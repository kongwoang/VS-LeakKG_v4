"""VS-LeakKG v3 — Knowledge Graph build pipeline.

Builds the leakage-detection knowledge graph from raw benchmark and
reference corpora. Audit downstream (contamination scoring, path
features, KG-NN diagnostics, figures) is deliberately not included —
those will be redesigned in a separate module against the KG outputs.

Pipeline tasks (sequential, idempotent — cached outputs are reused):

  1. load_chembl       ChEMBL ligands/assays/documents/targets
  2. load_bindingdb    BindingDB ligands/records
  3. chembl_map        benchmark <-> ChEMBL ligand map
  4. bindingdb_map     benchmark <-> BindingDB ligand map
  5. chembl_provenance per-mapped-molregno activity provenance
  6. load_bigbind      BigBind activities -> Examples/Ligands/Proteins
  7. build_kg          concat per-corpus + ChEMBL/BindingDB -> kg_*

Outputs land under:
  data/processed/   *.parquet (intermediate and final)
  outputs/reports/  *.md (per-task summaries)
  outputs/logs/     run + disk log
  outputs/reports/todos/ deferred-task notes

Run end-to-end:
  PYTHONPATH=src python -m vsleakkg.build_kg

Re-run a single task:
  PYTHONPATH=src python -c \\
    "from vsleakkg.build_kg import run_task, task_build_kg; \\
     run_task('build_kg', task_build_kg)"
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from vsleakkg import chem as vc
from vsleakkg import build_graph as vb
from vsleakkg import load_chembl_db, load_bigbind, load_bayesbind
from vsleakkg import load_dude, load_dekois, load_litpcba_ave


# -------- paths --------
ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES    = ROOT / "outputs" / "tables"
REPORTS   = ROOT / "outputs" / "reports"
LOGS      = ROOT / "outputs" / "logs"
TODOS     = REPORTS / "todos"
RUN_LOG   = LOGS / "kg_build.log"
DISK_LOG  = LOGS / "kg_build_disk.log"
STATUS_MD = REPORTS / "kg_build_status.md"

CHEMBL_DB     = RAW / "ChEMBL" / "extracted" / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db"
BINDINGDB_TSV = RAW / "BindingDB" / "extracted" / "BindingDB_All.tsv"
BIGBIND_META  = RAW / "BigBind" / "metadata" / "BigBindV1.5"
BIGBIND_EXTRACTED = RAW / "BigBind" / "extracted"
BAYESBIND_ROOT = RAW / "BayesBind" / "extracted"
DUDE_ROOT      = RAW / "DUD-E"
DEKOIS_ROOT    = RAW / "DEKOIS" / "extracted"          # contains DEKOIS2/<target>/
LITPCBA_AVE_ROOT = RAW / "LIT-PCBA" / "splits" / "AVE_unbiased"

for d in (PROCESSED, TABLES, REPORTS, LOGS, TODOS):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(RUN_LOG, mode="a", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vsleakkg.build_kg")


# -------- helpers --------
def ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_disk(event: str, target: str) -> None:
    lines = [f"==== {ts()} ====", f"event: {event}", f"target: {target}",
             f"cwd: {os.getcwd()}"]
    try:
        u = shutil.disk_usage(ROOT)
        lines.append(f"  free={u.free/1024**3:.2f}GB used={u.used/1024**3:.2f}GB")
    except OSError:
        pass
    DISK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISK_LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def append_status(task: str, status: str, note: str) -> None:
    if not STATUS_MD.exists():
        STATUS_MD.write_text("# VS-LeakKG v3 build status\n\n", encoding="utf-8")
    with open(STATUS_MD, "a", encoding="utf-8") as f:
        f.write(f"## {task} — **{status}** ({ts()})\n\n{note}\n\n")


def write_todo(task: str, body: str) -> None:
    p = TODOS / f"{task}.md"
    p.write_text(f"# {task} — manual action / blocker\n\n{ts()}\n\n{body}\n",
                 encoding="utf-8")


def run_task(name: str, fn: Callable[[], str]) -> bool:
    log.info("=== %s START ===", name)
    log_disk("task_start", name)
    t0 = time.time()
    try:
        note = fn() or "ok"
        dt = time.time() - t0
        append_status(name, "completed", f"{note}\n\nElapsed: {dt:.1f}s")
        log.info("=== %s OK (%.1fs) ===", name, dt)
        log_disk("task_end_ok", name)
        return True
    except Exception as exc:
        dt = time.time() - t0
        tb = traceback.format_exc()
        log.exception("=== %s FAILED ===", name)
        write_todo(name, f"```\n{tb}\n```\n\nElapsed before failure: {dt:.1f}s")
        append_status(name, "failed", f"{exc}\n\nSee `outputs/reports/todos/{name}.md`.")
        log_disk("task_end_fail", name)
        return False


# Edge types whose src is an example_id. An example_id carries its corpus
# (`ex:LIT-PCBA:VDR:2404447`), so these can NEVER be deduplicated across corpora and
# their counts must survive the merge EXACTLY. `ligand_has_scaffold` is deliberately
# absent: its src is a globally-shared Ligand, so a ligand in two corpora legitimately
# emits the same edge twice and the dedup legitimately removes one.
EXAMPLE_SCOPED_EDGES = (
    "example_has_ligand", "example_from_source", "example_targets_protein",
    "example_in_split", "example_has_label_type", "example_uses_decoy_protocol",
)


def expected_edge_counts() -> dict[str, int]:
    """Per-edge-type counts summed straight from the per-corpus parquets."""
    exp: dict[str, int] = {}
    for _, slug in (("LIT-PCBA-AVE", "litpcba_ave"), ("DUD-E", "dude"),
                    ("DEKOIS", "dekois"), ("BigBind", "bigbind"),
                    ("BayesBind", "bayesbind")):
        f = PROCESSED / f"{slug}_edges.parquet"
        if not f.exists():
            continue
        d = (pl.read_parquet(f, columns=["edge_type"])
               .group_by("edge_type").agg(pl.len().alias("n")))
        for et, n in d.iter_rows():
            exp[et] = exp.get(et, 0) + n
    return exp


def validate_raw_kg(nodes: pl.DataFrame, edges: pl.DataFrame,
                    expected_examples: int) -> None:
    """Refuse to write a corrupt RAW KG. Runs at the end of task_build_kg.

    `consolidate` grew a write-time guard after the null-byte incident; `build_kg`,
    where the corruption actually happens, never did. So a corrupt raw KG was written,
    and the ~1 h `ligand_similarity` pass then ran on top of it before anything
    complained. Observed on a bad run (2026-07-14 22:05): 1,128,590 NUL-corrupted
    `node_type` values, 361,112 corrupted labels, and `example_has_ligand` collapsed
    from 5,025,493 edges to 48,207 — a KG missing 99 % of its examples, written out
    without a murmur.

    Two classes of check, because NUL bytes are only the visible symptom:
      - no NUL byte in any string column;
      - the row counts that MUST hold by construction (one ligand and one source per
        Example) actually hold. A silent dedup collapse does not produce NUL bytes.

    A failure means the RUN is bad, not the code. Re-run.
    """
    problems: list[str] = []
    for frame, name in ((nodes, "nodes"), (edges, "edges")):
        for c, dt in zip(frame.columns, frame.dtypes):
            if dt != pl.Utf8:
                continue
            bad = frame.filter(pl.col(c).str.contains("\x00")).height
            if bad:
                problems.append(f"{name}.{c}: {bad:,} value(s) contain NUL bytes")

    # EXACT counts, not "roughly right". A run on 2026-07-14 lost 23,969 LIT-PCBA
    # `example_has_ligand` edges — no NUL bytes, no duplicate ids, both endpoints
    # present as nodes, and the minimal repro of concat().unique() over the same
    # parquets keeps every row. The loss is intermittent and it is SILENT: the only
    # thing that can catch it is comparing the output against the input, exactly.
    # A tolerance of 99 % would have waved it through — 5,001,524 / 5,025,493 = 99.52 %
    # — and 23,969 facts that were never recorded cannot be recovered anywhere.
    exp = expected_edge_counts()
    for et in EXAMPLE_SCOPED_EDGES:
        want = exp.get(et)
        if not want:
            continue
        got = edges.filter(pl.col("edge_type") == et).height
        if got != want:
            problems.append(
                f"{et}: {got:,} edges, the per-corpus parquets hold {want:,} "
                f"({got - want:+,}). These are example-scoped — they CANNOT dedup "
                f"across corpora, so any difference is lost data.")

    n_ex = nodes.filter(pl.col("node_type") == "Example").height
    if n_ex != expected_examples:
        problems.append(
            f"Example nodes: {n_ex:,}, expected exactly {expected_examples:,} "
            f"({n_ex - expected_examples:+,})")

    # Every benchmark Ligand that the map resolves to a ChEMBL compound must actually
    # CARRY the link edge — that edge is the only route by which an Example reaches an
    # Assay or a Publication. A dedup keyed on molregno instead of on the pair silently
    # dropped 2,809 of them, and with them the entire provenance of 8,811 Examples.
    # Nothing failed; the edges simply were not there.
    f = PROCESSED / "benchmark_to_chembl_ligand_map.parquet"
    if f.exists():
        want = (pl.read_parquet(f, columns=["canonical_smiles", "molregno"])
                  .filter(pl.col("molregno").is_not_null())["canonical_smiles"].n_unique())
        got = edges.filter(
            pl.col("edge_type") == "benchmark_ligand_same_inchikey_as_chembl_ligand"
        )["src"].n_unique()
        if got != want:
            problems.append(
                f"benchmark->ChEMBL link: {got:,} Ligands carry the edge, but the map "
                f"resolves {want:,} ({got - want:+,}). Those Ligands' Examples reach no "
                f"Assay and no Publication.")

    if problems:
        raise RuntimeError(
            "raw KG failed write-time validation — REFUSING to write.\n  "
            + "\n  ".join(problems)
            + "\n\nThis is the intermittent polars string-corruption / dedup-collapse "
              "bug that this box hits under load. The RUN is bad, not the code. "
              "Re-run build_kg (consider POLARS_MAX_THREADS=8)."
        )


# Ligand node id from canonical SMILES (used everywhere KG-side).
def _mhash(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()


def _lig_node_id(canon: str) -> str:
    return f"lig:{_mhash(canon)}"


# -------- 1. load_chembl --------
def task_load_chembl() -> str:
    if not CHEMBL_DB.exists():
        raise FileNotFoundError(CHEMBL_DB)
    log.info("ChEMBL DB at %s", CHEMBL_DB)
    out_lig = PROCESSED / "chembl_ligands.parquet"
    out_tgt = PROCESSED / "chembl_targets.parquet"
    out_doc = PROCESSED / "chembl_documents.parquet"
    out_asy = PROCESSED / "chembl_assays.parquet"

    out_acc = PROCESSED / "chembl_target_accessions.parquet"

    conn = load_chembl_db.connect(CHEMBL_DB)
    if not out_lig.exists():
        log.info("ChEMBL: loading ligands ...")
        load_chembl_db.load_ligands(conn).write_parquet(out_lig)
    if not out_tgt.exists():
        load_chembl_db.load_targets(conn).write_parquet(out_tgt)
    if not out_acc.exists():
        # ChEMBL target -> UniProt accession. `load_target_sequences` has always
        # returned this and nothing ever used it, so the KG had no way to turn
        # "activity measured against ChEMBL target CHEMBL204" into "this ligand was
        # measured against protein P00734". Consequence: `ligand_measured_protein` —
        # documented as "a ChEMBL/BindingDB-trained model has seen this (ligand,
        # protein) pair" — was BindingDB-only. BindingDB contributed 378,427 pairs
        # over 140,457 ligands; ChEMBL, the dominant pretraining corpus, contributed
        # ZERO, and had 3,436,257 pairs over 604,978 ligands to give.
        log.info("ChEMBL: extracting target -> UniProt accession map ...")
        (load_chembl_db.load_target_sequences(conn)
            .select("target_chembl_id", "accession")
            .drop_nulls().unique()
            .write_parquet(out_acc))
    if not out_doc.exists():
        load_chembl_db.load_documents(conn).write_parquet(out_doc)
    if not out_asy.exists():
        load_chembl_db.load_assays(conn).write_parquet(out_asy)
    conn.close()

    n_lig = pl.read_parquet(out_lig).height
    n_tgt = pl.read_parquet(out_tgt).height
    n_doc = pl.read_parquet(out_doc).height
    n_asy = pl.read_parquet(out_asy).height
    (REPORTS / "chembl_processed_tables_report.md").write_text(
        "# ChEMBL processed tables\n\n" + ts() + "\n\n"
        f"- `chembl_ligands.parquet`:    {n_lig:,} rows\n"
        f"- `chembl_targets.parquet`:    {n_tgt:,} rows\n"
        f"- `chembl_documents.parquet`:  {n_doc:,} rows\n"
        f"- `chembl_assays.parquet`:     {n_asy:,} rows\n\n"
        "Activities are pulled on demand by `chembl_provenance` task only for\n"
        "molregnos that map from benchmark ligands.\n",
        encoding="utf-8")
    return f"ligands={n_lig:,} targets={n_tgt:,} docs={n_doc:,} assays={n_asy:,}"


# -------- 2. load_bindingdb --------
def task_load_bindingdb() -> str:
    if not BINDINGDB_TSV.exists():
        raise FileNotFoundError(BINDINGDB_TSV)
    lig_out = PROCESSED / "bindingdb_ligands_minimal.parquet"
    rec_out = PROCESSED / "bindingdb_records_minimal.parquet"
    if lig_out.exists() and rec_out.exists():
        return f"cached lig={pl.read_parquet(lig_out).height:,} rec={pl.read_parquet(rec_out).height:,}"

    with open(BINDINGDB_TSV, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    col_idx = {c: i for i, c in enumerate(header)}
    NEEDED = {
        "Ligand SMILES": "ligand_smiles",
        "Ligand InChI Key": "ligand_inchikey",
        "Target Name": "target_name",
        "Ki (nM)": "ki_nM",
        "IC50 (nM)": "ic50_nM",
        "Kd (nM)": "kd_nM",
        "EC50 (nM)": "ec50_nM",
        "Article DOI": "article_doi",
        "PMID": "pmid",
        "PubChem AID": "pubchem_aid",
        "PubChem CID": "pubchem_cid",
        "ChEMBL ID of Ligand": "chembl_id_ligand",
        "ZINC ID of Ligand": "zinc_id_ligand",
        "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_swissprot_id",
        "UniProt (SwissProt) Recommended Name of Target Chain 1": "uniprot_name",
        "Target Source Organism According to Curator or DataSource": "target_organism",
        "BindingDB Reactant_set_id": "bindingdb_record_id",
    }
    idxs = {NEEDED[c]: col_idx[c] for c in NEEDED if c in col_idx}
    out_cols = list(idxs.keys())

    log.info("BindingDB: streaming TSV with %d cols of interest", len(out_cols))
    flat: List[list] = []
    lig_seen: Dict[str, list] = {}
    n_rows = 0
    max_idx = max(idxs.values())
    with open(BINDINGDB_TSV, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max_idx:
                continue
            row = [parts[idxs[c]] for c in out_cols]
            flat.append(row)
            ik = row[out_cols.index("ligand_inchikey")] if "ligand_inchikey" in out_cols else ""
            if ik and ik not in lig_seen:
                lig_seen[ik] = [
                    row[out_cols.index("ligand_smiles")] if "ligand_smiles" in out_cols else "",
                    row[out_cols.index("chembl_id_ligand")] if "chembl_id_ligand" in out_cols else "",
                    row[out_cols.index("zinc_id_ligand")] if "zinc_id_ligand" in out_cols else "",
                    row[out_cols.index("pubchem_cid")] if "pubchem_cid" in out_cols else "",
                ]
            n_rows += 1
            if n_rows % 500_000 == 0:
                log.info("BindingDB rows read: %d", n_rows)

    rec_df = pl.DataFrame(flat, schema=out_cols, orient="row")
    rec_df.write_parquet(rec_out)
    log.info("BindingDB: %d records written", rec_df.height)

    lig_rows = [(ik, *v) for ik, v in lig_seen.items()]
    lig_df = pl.DataFrame(
        lig_rows,
        schema=["ligand_inchikey", "ligand_smiles", "chembl_id_ligand",
                "zinc_id_ligand", "pubchem_cid"],
        orient="row",
    )
    lig_df.write_parquet(lig_out)
    log.info("BindingDB: %d unique ligands written", lig_df.height)
    return f"records={rec_df.height:,}, unique_ligands={lig_df.height:,}"


# -------- 3. chembl_map --------
# Per-corpus inputs to the benchmark <-> reference DB joins.
# After v3 redesign (PDBBind dropped, BigBind added) the corpus list is:
_CORPORA_FOR_MAPPING = (
    ("LIT-PCBA AVE", "litpcba_ave_examples.parquet", "smiles_canonical", "inchikey"),
    ("DUD-E",        "dude_examples.parquet",        "smiles_canonical", "inchikey"),
    ("DEKOIS",       "dekois_examples.parquet",      "smiles_canonical", "inchikey"),
    ("BigBind",      "bigbind_examples.parquet",     "smiles_canonical", "inchikey"),
    ("BayesBind",    "bayesbind_examples.parquet",   "smiles_canonical", "inchikey"),
)


def task_chembl_map() -> str:
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet")
    chembl_ik = (chembl_lig.filter(pl.col("standard_inchi_key").is_not_null())
                 .group_by("standard_inchi_key")
                 .agg([pl.col("molregno").first().alias("molregno"),
                       pl.col("molecule_chembl_id").first().alias("molecule_chembl_id"),
                       pl.len().alias("n_chembl_rows")])
                 .rename({"standard_inchi_key": "inchikey"}))
    chembl_smi = (chembl_lig.filter(pl.col("canonical_smiles").is_not_null())
                  .group_by("canonical_smiles")
                  .agg([pl.col("molregno").first().alias("molregno_smi"),
                        pl.col("molecule_chembl_id").first().alias("molecule_chembl_id_smi")]))

    parts = []
    sources = []
    for ds, fname, smi, ik in _CORPORA_FOR_MAPPING:
        parq = PROCESSED / fname
        if not parq.exists():
            log.warning("chembl_map: skip %s (parquet not present)", ds)
            continue
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi).alias("canonical_smiles"),
                       pl.col(ik).alias("inchikey")])
              .filter(pl.col("canonical_smiles").is_not_null())
              .unique()
              .with_columns(pl.lit(ds).alias("benchmark_dataset"))
              .collect())
        j = df.join(chembl_ik, on="inchikey", how="left") \
              .join(chembl_smi, on="canonical_smiles", how="left") \
              .with_columns([
                  pl.coalesce(["molregno", "molregno_smi"]).alias("molregno"),
                  pl.coalesce(["molecule_chembl_id", "molecule_chembl_id_smi"]).alias("molecule_chembl_id"),
                  pl.when(pl.col("molregno").is_not_null()).then(pl.lit("inchikey"))
                    .when(pl.col("molregno_smi").is_not_null()).then(pl.lit("canonical_smiles"))
                    .otherwise(pl.lit("unmatched")).alias("match_method"),
              ]).select(["benchmark_dataset", "canonical_smiles", "inchikey",
                         "molregno", "molecule_chembl_id", "match_method"])
        parts.append(j)
        n = df.height
        n_m = int((j["match_method"] != "unmatched").sum())
        sources.append((ds, n, n_m, n_m / n if n else 0.0))
        log.info("chembl_map: %s -> %d / %d (%.2f%%)", ds, n_m, n, 100*n_m/n if n else 0)
    out = pl.concat(parts, how="diagonal_relaxed")
    out.write_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")

    body = "# Benchmark -> ChEMBL ligand mapping\n\n" + ts() + "\n\n"
    body += "| benchmark | unique ligands | mapped | rate |\n|---|---:|---:|---:|\n"
    for ds, n, nm, r in sources:
        body += f"| {ds} | {n:,} | {nm:,} | {r:.2%} |\n"
    body += "\nMapping priority: exact InChIKey, then canonical SMILES. No fuzzy.\n"
    (REPORTS / "benchmark_to_chembl_mapping_report.md").write_text(body, encoding="utf-8")
    return ", ".join(f"{ds}={nm:,}/{n:,}" for ds, n, nm, _ in sources)


# -------- 4. bindingdb_map --------
def task_bindingdb_map() -> str:
    bdb = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")
    bdb_ik = (bdb.filter(pl.col("ligand_inchikey").is_not_null() & (pl.col("ligand_inchikey") != ""))
              .unique(subset=["ligand_inchikey"]))

    parts = []
    sources = []
    for ds, fname, smi, ik in _CORPORA_FOR_MAPPING:
        parq = PROCESSED / fname
        if not parq.exists():
            log.warning("bindingdb_map: skip %s (parquet not present)", ds)
            continue
        df = (pl.scan_parquet(parq)
              .select([pl.col(smi).alias("canonical_smiles"),
                       pl.col(ik).alias("inchikey")])
              .filter(pl.col("canonical_smiles").is_not_null())
              .unique()
              .with_columns(pl.lit(ds).alias("benchmark_dataset"))
              .collect())
        j = df.join(bdb_ik.rename({"ligand_inchikey": "inchikey"}),
                    on="inchikey", how="left")
        bdb_smi = (bdb.filter(pl.col("ligand_smiles").is_not_null() & (pl.col("ligand_smiles") != ""))
                   .unique(subset=["ligand_smiles"]))
        j = j.join(bdb_smi.rename({"ligand_smiles": "canonical_smiles_smi_match",
                                   "ligand_inchikey": "inchikey_smi"}),
                   left_on="canonical_smiles", right_on="canonical_smiles_smi_match",
                   how="left") \
            .with_columns([
                pl.when(pl.col("ligand_smiles").is_not_null()).then(pl.lit("inchikey"))
                  .when(pl.col("inchikey_smi").is_not_null()).then(pl.lit("canonical_smiles"))
                  .otherwise(pl.lit("unmatched")).alias("match_method"),
            ]) \
            .select(["benchmark_dataset", "canonical_smiles", "inchikey", "match_method"])
        parts.append(j)
        n = df.height
        n_m = int((j["match_method"] != "unmatched").sum())
        sources.append((ds, n, n_m, n_m / n if n else 0.0))
        log.info("bindingdb_map: %s -> %d / %d (%.2f%%)", ds, n_m, n, 100*n_m/n if n else 0)
    res = pl.concat(parts, how="diagonal_relaxed")
    res.write_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    body = "# Benchmark -> BindingDB ligand mapping\n\n" + ts() + "\n\n"
    body += "| benchmark | unique ligands | mapped | rate |\n|---|---:|---:|---:|\n"
    for ds, n, nm, r in sources:
        body += f"| {ds} | {n:,} | {nm:,} | {r:.2%} |\n"
    (REPORTS / "benchmark_to_bindingdb_mapping_report.md").write_text(body, encoding="utf-8")
    return ", ".join(f"{ds}={nm:,}/{n:,}" for ds, n, nm, _ in sources)


# -------- 5. chembl_provenance --------
def task_chembl_provenance() -> str:
    mp = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mapped = mp.filter(pl.col("molregno").is_not_null()).unique(subset=["molregno"])
    molregnos = [int(m) for m in mapped["molregno"].to_list()]
    log.info("chembl_provenance: pulling activities for %d unique molregnos", len(molregnos))
    conn = load_chembl_db.connect(CHEMBL_DB)
    acts = load_chembl_db.load_activities_for_molregnos(conn, molregnos)
    conn.close()
    log.info("chembl_provenance: %d activities pulled", acts.height)

    assays = pl.read_parquet(PROCESSED / "chembl_assays.parquet")
    docs = pl.read_parquet(PROCESSED / "chembl_documents.parquet")
    targets = pl.read_parquet(PROCESSED / "chembl_targets.parquet")

    enriched = (acts
        .join(assays, on="assay_id", how="left")
        .join(docs,   left_on="doc_id", right_on="doc_id", how="left")
        .join(targets, on="tid", how="left")) \
        .with_columns([
            pl.when(pl.col("target_chembl_id").is_not_null() & pl.col("document_chembl_id").is_not_null())
              .then(pl.lit("ligand_target_assay_document"))
              .when(pl.col("document_chembl_id").is_not_null())
              .then(pl.lit("ligand_assay_document"))
              .when(pl.col("assay_chembl_id").is_not_null())
              .then(pl.lit("ligand_assay"))
              .otherwise(pl.lit("ligand_only"))
              .alias("provenance_level"),
            pl.lit("candidate").alias("confidence"),
        ])

    benchmark_prov = (mapped.join(enriched, on="molregno", how="left")
                      .select([
                          "benchmark_dataset", "canonical_smiles", "inchikey",
                          "molregno", "molecule_chembl_id",
                          "activity_id", "assay_id", "assay_chembl_id",
                          "doc_id", "document_chembl_id",
                          "target_chembl_id",
                          "standard_type", "standard_relation",
                          "standard_value", "standard_units", "pchembl_value",
                          "provenance_level", "confidence",
                      ]))
    benchmark_prov.write_parquet(PROCESSED / "benchmark_chembl_candidate_provenance.parquet")
    log.info("chembl_provenance: wrote %d rows", benchmark_prov.height)

    by_level = (benchmark_prov.group_by("provenance_level").agg(pl.len().alias("n"))
                .sort("n", descending=True))
    (REPORTS / "chembl_candidate_provenance_report.md").write_text(
        "# ChEMBL candidate provenance\n\n" + ts() + "\n\n"
        f"- mapped molregnos: **{len(molregnos):,}**\n"
        f"- activity rows pulled: **{acts.height:,}**\n"
        f"- benchmark-provenance rows: **{benchmark_prov.height:,}**\n\n"
        "## Counts by provenance level\n\n"
        + by_level.to_pandas().to_string(index=False) + "\n",
        encoding="utf-8")
    return f"prov_rows={benchmark_prov.height:,} mapped_molregnos={len(molregnos):,}"


# -------- 6. load_bayesbind --------
def task_load_bayesbind() -> str:
    """Parse BayesBind per-target actives/random CSVs into per-corpus parquets."""
    if not BAYESBIND_ROOT.exists():
        raise FileNotFoundError(BAYESBIND_ROOT)
    out_ex  = PROCESSED / "bayesbind_examples.parquet"
    out_n   = PROCESSED / "bayesbind_nodes.parquet"
    out_e   = PROCESSED / "bayesbind_edges.parquet"
    if out_ex.exists() and out_n.exists() and out_e.exists():
        return (f"cached examples={pl.read_parquet(out_ex).height:,} "
                f"nodes={pl.read_parquet(out_n).height:,} "
                f"edges={pl.read_parquet(out_e).height:,}")
    examples, nodes, edges = load_bayesbind.build(
        extracted_dir=BAYESBIND_ROOT, log=log,
    )
    examples.write_parquet(out_ex)
    nodes.write_parquet(out_n)
    edges.write_parquet(out_e)
    (REPORTS / "bayesbind_loader_report.md").write_text(
        "# BayesBind loader summary\n\n" + ts() + "\n\n"
        f"- examples: {examples.height:,}\n"
        f"- nodes:    {nodes.height:,}\n"
        f"- edges:    {edges.height:,}\n",
        encoding="utf-8")
    return f"examples={examples.height:,} nodes={nodes.height:,} edges={edges.height:,}"


# -------- 7. load_bigbind --------
def task_load_bigbind() -> str:
    """Parse BigBind activities + structures CSVs; emit per-corpus parquets.

    Inputs:  data/raw/BigBind/metadata/BigBindV1.5/{activities,structures}_*.csv
             data/raw/BigBind/extracted/...                (optional, for SDF/PDB)
    Outputs: data/processed/bigbind_examples.parquet  (one row per activity)
             data/processed/bigbind_nodes.parquet     (Ligand, Protein, Example, ...)
             data/processed/bigbind_edges.parquet     (example_has_ligand, _has_protein, _from_source, ligand_has_scaffold, ...)
    """
    if not BIGBIND_META.exists():
        raise FileNotFoundError(BIGBIND_META)
    out_ex  = PROCESSED / "bigbind_examples.parquet"
    out_n   = PROCESSED / "bigbind_nodes.parquet"
    out_e   = PROCESSED / "bigbind_edges.parquet"
    if out_ex.exists() and out_n.exists() and out_e.exists():
        return (f"cached examples={pl.read_parquet(out_ex).height:,} "
                f"nodes={pl.read_parquet(out_n).height:,} "
                f"edges={pl.read_parquet(out_e).height:,}")

    examples, nodes, edges = load_bigbind.build(
        meta_dir=BIGBIND_META,
        extracted_dir=BIGBIND_EXTRACTED if BIGBIND_EXTRACTED.exists() else None,
        log=log,
    )
    examples.write_parquet(out_ex)
    nodes.write_parquet(out_n)
    edges.write_parquet(out_e)
    (REPORTS / "bigbind_loader_report.md").write_text(
        "# BigBind loader summary\n\n" + ts() + "\n\n"
        f"- examples: {examples.height:,}\n"
        f"- nodes:    {nodes.height:,}\n"
        f"- edges:    {edges.height:,}\n",
        encoding="utf-8")
    return f"examples={examples.height:,} nodes={nodes.height:,} edges={edges.height:,}"


# -------- 6c. the three benchmark corpora --------
# These loaders existed in `src/` but were NOT wired into TASKS. `task_build_kg`
# reads data/processed/<slug>_{nodes,edges}.parquet and merely log.warning()s
# when a corpus is missing, so a clean checkout produced a KG with BigBind +
# BayesBind only — 843,833 of 5,025,493 examples — and reported success. The
# DUD-E / DEKOIS / LIT-PCBA parquets on disk were inherited artefacts whose
# generating code was not in this pipeline: 83 % of the graph could not be
# rebuilt from the repo. Wiring them back is the difference between a KG we can
# defend and one we merely possess.


def _featurize_corpus(df: pl.DataFrame, human: str) -> pl.DataFrame:
    """Add smiles_canonical / inchikey / scaffold_smiles / parse_ok.

    Rows that fail to parse are KEPT, unlike the BigBind loader which drops
    them. The inherited DUD-E parquet has 1,434,019 Examples but only 1,434,015
    example_has_ligand edges: the 4 unparseable rows still became ligand-less
    Example nodes, and `fixes.drop_ligandless_examples` removes them during
    consolidation. Dropping them here would shift `_row_idx` and therefore
    change every example_id downstream of the bad row.
    """
    feats = vc.featurize_batch_parallel(df["smiles_input"].to_list(), log=log)
    out = df.with_columns([
        pl.Series("smiles_canonical", [f.smiles_canonical for f in feats]),
        pl.Series("inchikey",         [f.inchikey for f in feats]),
        pl.Series("scaffold_smiles",  [f.scaffold_smiles for f in feats]),
        pl.Series("parse_ok",         [f.parse_ok for f in feats]),
    ])
    n_bad = int((~out["parse_ok"]).sum())
    if n_bad:
        log.warning("%s: %d rows failed RDKit parse — kept as ligand-less Examples",
                    human, n_bad)
    return out


def _build_corpus(slug: str, human: str, examples_df: pl.DataFrame,
                  *, include_decoy_protocol: bool) -> str:
    """Featurize + emit <slug>_{examples,nodes,edges}.parquet."""
    df = _featurize_corpus(examples_df, human)
    examples = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        examples,
        include_decoy_protocol=include_decoy_protocol,
        include_protein_target=True,
    )
    df.write_parquet(PROCESSED / f"{slug}_examples.parquet")
    nodes.write_parquet(PROCESSED / f"{slug}_nodes.parquet")
    edges.write_parquet(PROCESSED / f"{slug}_edges.parquet")
    log.info("%s: %d examples, %d nodes, %d edges",
             human, df.height, nodes.height, edges.height)
    return f"examples={df.height:,} nodes={nodes.height:,} edges={edges.height:,}"


def _cached(slug: str) -> str | None:
    ex = PROCESSED / f"{slug}_examples.parquet"
    n  = PROCESSED / f"{slug}_nodes.parquet"
    e  = PROCESSED / f"{slug}_edges.parquet"
    if ex.exists() and n.exists() and e.exists():
        return (f"cached examples={pl.read_parquet(ex).height:,} "
                f"nodes={pl.read_parquet(n).height:,} "
                f"edges={pl.read_parquet(e).height:,}")
    return None


def task_load_dude() -> str:
    """DUD-E: 102 targets, actives_final.ism + decoys_final.ism per target."""
    if (c := _cached("dude")):
        return c
    if not DUDE_ROOT.exists():
        raise FileNotFoundError(DUDE_ROOT)
    # Decoys are generated (property-matched), so the corpus gets a protocol node.
    return _build_corpus("dude", "DUD-E", load_dude.load_all(DUDE_ROOT),
                         include_decoy_protocol=True)


def task_load_dekois() -> str:
    """DEKOIS 2.0: one active_decoys.smi per target; BDB* = active, ZINC* = decoy."""
    if (c := _cached("dekois")):
        return c
    if not DEKOIS_ROOT.exists():
        raise FileNotFoundError(DEKOIS_ROOT)
    return _build_corpus("dekois", "DEKOIS", load_dekois.load_all(DEKOIS_ROOT),
                         include_decoy_protocol=True)


def task_load_litpcba() -> str:
    """LIT-PCBA, AVE_unbiased splits: 15 targets, active/inactive x train/validation.

    No decoy-protocol node: LIT-PCBA inactives are *experimentally measured*, not
    generated. Linking them through a shared protocol would fabricate a leakage
    path that does not exist (see fixes.DECOY_PROTOCOL).
    """
    if (c := _cached("litpcba_ave")):
        return c
    if not LITPCBA_AVE_ROOT.exists():
        raise FileNotFoundError(LITPCBA_AVE_ROOT)
    return _build_corpus("litpcba_ave", "LIT-PCBA",
                         load_litpcba_ave.load_all(LITPCBA_AVE_ROOT),
                         include_decoy_protocol=False)


# -------- 7. build_kg --------
def task_build_kg() -> str:
    """Build the final KG by concatenating per-corpus parquets and adding the
    ChEMBL/BindingDB cross-reference layer.

    Inputs:  data/processed/{litpcba_ave,dude,dekois,bigbind}_{nodes,edges,examples}.parquet
             data/processed/{chembl_ligands,bindingdb_ligands_minimal}.parquet
             data/processed/benchmark_chembl_candidate_provenance.parquet
             data/processed/benchmark_to_{chembl,bindingdb}_ligand_map.parquet
    Outputs: data/processed/{kg_nodes,kg_edges}.parquet
             outputs/reports/kg_build_summary.md
    """
    CORPORA = [
        ("LIT-PCBA-AVE", "litpcba_ave"),
        ("DUD-E",        "dude"),
        ("DEKOIS",       "dekois"),
        ("BigBind",      "bigbind"),
        ("BayesBind",    "bayesbind"),
    ]
    base_n_parts: list = []
    base_e_parts: list = []
    loaded: list = []
    missing: list[str] = []
    for human, slug in CORPORA:
        n_path = PROCESSED / f"{slug}_nodes.parquet"
        e_path = PROCESSED / f"{slug}_edges.parquet"
        if n_path.exists() and e_path.exists():
            base_n_parts.append(pl.read_parquet(n_path))
            base_e_parts.append(pl.read_parquet(e_path))
            loaded.append(human)
        else:
            missing.append(human)
    # A missing corpus used to be a log.warning() — so the build happily produced
    # a KG covering 17 % of the examples and called itself complete. A KG that is
    # silently missing four fifths of the corpus is worse than no KG: every
    # leakage number computed on it is wrong and nothing says so.
    if missing:
        raise RuntimeError(
            f"per-corpus parquets missing for: {', '.join(missing)}. "
            f"Run the corresponding load_* task first — do NOT build a partial KG.")
    # rechunk() before unique(): the corruption shows up as string buffers zeroed out
    # while their offsets survive (node_ids become runs of NUL bytes of exactly the
    # right length). Concatenating five multi-million-row frames leaves the string
    # column in many chunks, and it is the multi-chunk path that goes wrong. Forcing a
    # single contiguous buffer first is not a proof, but it removes the trigger we can
    # actually see. The write-time guards below are the real protection.
    base_n = (pl.concat(base_n_parts, how="vertical_relaxed")
                .rechunk().unique(subset=["node_id"]))
    base_e = pl.concat(base_e_parts, how="vertical_relaxed").rechunk().unique()
    for _f, _c, _w in ((base_n, "node_id", "nodes"), (base_e, "edge_type", "edges")):
        _k = _f.filter(pl.col(_c).str.contains("\x00")).height
        if _k:
            raise RuntimeError(
                f"CORRUPTION at the per-corpus concat: {_k:,} {_w}.{_c} values hold "
                f"NUL bytes, though every input parquet is clean. Re-run.")
    log.info("KG base after per-corpus dedup: %d nodes %d edges (from %s)",
             base_n.height, base_e.height, "+".join(loaded))

    nodes_new: List[tuple] = []
    edges_new: List[tuple] = []

    # ---- Cross-corpus same_inchikey_as + same_parent_inchikey_as edges ----
    # same_inchikey_as: same InChIKey, different canonical SMILES (tautomer /
    #   stereo). lig:md5(canonical) IDs don't collapse these.
    # same_parent_inchikey_as: same salt-stripped parent InChIKey but different
    #   full InChIKey (HCl salt vs free base, protonation states). Addresses
    #   the 178K salt-drift cases surfaced by merge_audit.
    # Accumulate (smi, inchikey) across all corpora first, then run parent
    # InChIKey computation in one parallel batch (32× speedup on VUW).
    smi_to_lig: dict = {}
    smi_to_ik: dict = {}
    ik_to_smis: dict = {}
    parent_to_smis: dict = {}
    all_unique_smis: list[str] = []  # preserve insertion order
    _seen_smi: set[str] = set()
    for human, slug in CORPORA:
        ex_path = PROCESSED / f"{slug}_examples.parquet"
        if not ex_path.exists():
            continue
        cols = pl.read_parquet_schema(ex_path)
        smi_col = "smiles_canonical" if "smiles_canonical" in cols else (
            "canonical_smiles" if "canonical_smiles" in cols else None)
        if smi_col is None or "inchikey" not in cols:
            continue
        df = (pl.scan_parquet(ex_path)
              .select([pl.col(smi_col).alias("smi"), pl.col("inchikey")])
              .filter(pl.col("smi").is_not_null() & pl.col("inchikey").is_not_null())
              .unique()
              .collect())
        smi_list = df["smi"].to_list()
        ik_list = df["inchikey"].to_list()
        for j in range(len(smi_list)):
            smi = smi_list[j]
            ik = ik_list[j]
            smi_to_lig.setdefault(smi, _lig_node_id(smi))
            smi_to_ik.setdefault(smi, ik)
            ik_to_smis.setdefault(ik, set()).add(smi)
            if smi not in _seen_smi:
                _seen_smi.add(smi)
                all_unique_smis.append(smi)
        log.info("  %s: %d unique SMILES (running total %d)",
                 human, len(smi_list), len(all_unique_smis))

    # Parallel parent InChIKey compute — order-preserving via imap, length-checked.
    log.info("computing parent InChIKey for %d unique SMILES (parallel) ...",
             len(all_unique_smis))
    parent_iks = vc.parent_inchikey_batch_parallel(all_unique_smis, log=log)
    assert len(parent_iks) == len(all_unique_smis), \
        f"parent_inchikey list length mismatch: {len(parent_iks)} vs {len(all_unique_smis)}"
    n_pik_computed = 0
    for j, smi in enumerate(all_unique_smis):
        pik = parent_iks[j]
        if pik:
            parent_to_smis.setdefault(pik, set()).add(smi)
            n_pik_computed += 1
    log.info("parent InChIKey: %d computed, %d distinct parents",
             n_pik_computed, len(parent_to_smis))
    cross_src = 0
    for ik, smis in ik_to_smis.items():
        if len(smis) <= 1:
            continue
        smis_list = sorted(smis)
        anchor = smi_to_lig[smis_list[0]]
        for s in smis_list[1:]:
            other = smi_to_lig[s]
            edges_new.append((anchor, other, "same_inchikey_as",
                              json.dumps({"inchikey": ik})))
            cross_src += 1
    log.info("cross-corpus same_inchikey_as edges: %d", cross_src)

    # same_parent_inchikey_as must carry information that same_inchikey_as does not.
    # It used to be emitted for EVERY parent group, including groups whose members
    # already share a full InChIKey — so it was a strict superset of same_inchikey_as
    # by construction, and on the shipped KG it came out an EXACT duplicate of it
    # (6,939 == 6,939 edges, identical pairs). Skipping the same-full-InChIKey pairs
    # makes the relation mean what its name says: same compound, DIFFERENT full key
    # (salt / protonation / stereo variant).
    cross_parent = 0
    skipped_same_ik = 0
    for pik, smis in parent_to_smis.items():
        if len(smis) <= 1:
            continue
        smis_list = sorted(smis)
        anchor = smis_list[0]
        anchor_id = smi_to_lig[anchor]
        for s in smis_list[1:]:
            if smi_to_ik.get(s) == smi_to_ik.get(anchor):
                skipped_same_ik += 1   # already bridged by same_inchikey_as
                continue
            edges_new.append((anchor_id, smi_to_lig[s], "same_parent_inchikey_as",
                              json.dumps({"parent_inchikey": pik})))
            cross_parent += 1
    log.info("cross-corpus same_parent_inchikey_as edges: %d (%d pairs skipped: "
             "already same full InChIKey)", cross_parent, skipped_same_ik)

    # ---- DatasetSource + DatabaseRelease nodes ----
    for src, release in (("ChEMBL35", "ChEMBL_35"),
                          ("BindingDB202605", "BindingDB_2026_05")):
        nodes_new.append((f"src:{src}", "DatasetSource", src, "{}"))
        nodes_new.append((f"dbrel:{release}", "DatabaseRelease", release, "{}"))

    # ---- ChEMBL ligand + activity + assay + document + target subgraph ----
    mp_chembl = pl.read_parquet(PROCESSED / "benchmark_to_chembl_ligand_map.parquet")
    mp_ok = mp_chembl.filter(pl.col("molregno").is_not_null())
    chembl_lig = pl.read_parquet(PROCESSED / "chembl_ligands.parquet")
    prov_path = PROCESSED / "benchmark_chembl_candidate_provenance.parquet"
    prov = pl.read_parquet(prov_path) if prov_path.exists() else pl.DataFrame()

    # unique(subset=["molregno"]) — one row per ChEMBL compound — was wrong, and it
    # was wrong in the direction that loses facts.
    #
    # The loop below emits TWO things per row: a ChEMBLLigand node (keyed on the ChEMBL
    # id, so deduping it is harmless — the node dedup handles that anyway) and the edge
    # `benchmark_ligand_same_inchikey_as_chembl_ligand`, which is keyed on the BENCHMARK
    # ligand. Two benchmark ligands can share one molregno: same compound, different
    # canonical SMILES (a tautomer or stereo variant — exactly the pairs `ligand_exact`
    # exists to bridge). Deduping on molregno kept one of them and silently dropped the
    # edge for the other, and that edge is the ONLY route by which an Example reaches
    # ChEMBL: no link, no `example_from_assay`, no `example_from_publication`.
    #
    # Measured on the shipped KG: 611,471 benchmark SMILES map onto 608,662 molregnos,
    # so 2,809 benchmark ligands never got the edge — 25,318 Examples ride on them and
    # 8,811 of those carry no assay edge at all. They are not missing provenance; the
    # provenance was thrown away.
    #
    # Key on the pair, which is what the edge is.
    mapped_mol = (mp_ok.join(chembl_lig.select(["molregno", "molecule_chembl_id",
                                                  "canonical_smiles", "standard_inchi_key"]),
                              on="molregno", how="left")
                  .unique(subset=["molregno", "canonical_smiles"]))
    # Pull columns to Python lists before the loop. iter_rows(named=True) over
    # large polars DataFrames (>5M rows in the chembl_provenance case) corrupts
    # strings — f-string interpolation returns null-byte-filled strings of the
    # expected length instead of the actual text. Working off plain lists avoids
    # the bug entirely.
    mm_molregno = mapped_mol["molregno"].to_list()
    mm_chid = mapped_mol["molecule_chembl_id"].to_list()
    mm_smi_right = mapped_mol["canonical_smiles_right"].to_list() if "canonical_smiles_right" in mapped_mol.columns else [None] * mapped_mol.height
    mm_smi_left = mapped_mol["canonical_smiles"].to_list()
    mm_ik = mapped_mol["standard_inchi_key"].to_list()
    mm_match = mapped_mol["match_method"].to_list()
    for i in range(mapped_mol.height):
        chid = mm_chid[i]
        if chid is None:
            continue
        nid = f"chembl_lig:{chid}"
        # mm_smi_left = canonical_smiles from benchmark_to_chembl_ligand_map,
        # which comes from the benchmark _examples parquet (RDKit-canonical
        # via the same vsleakkg.chem.featurize pipeline as the per-corpus
        # Ligand nodes). Use THIS for the benchmark_lid lookup so the edge
        # lands on the right Ligand node. Store the ChEMBL-side canonical
        # in the ChEMBLLigand props for traceability.
        bench_smi = mm_smi_left[i]
        chembl_smi = mm_smi_right[i]
        nodes_new.append((nid, "ChEMBLLigand", chid,
                          json.dumps({"molregno": int(mm_molregno[i]) if mm_molregno[i] is not None else None,
                                       "canonical_smiles_chembl": chembl_smi,
                                       "canonical_smiles_benchmark": bench_smi,
                                       "inchikey": mm_ik[i]})))
        edges_new.append((nid, "src:ChEMBL35", "chembl_ligand_from_source", "{}"))
        if bench_smi:
            benchmark_lid = _lig_node_id(bench_smi)
            edges_new.append((benchmark_lid, nid,
                              "benchmark_ligand_same_inchikey_as_chembl_ligand",
                              json.dumps({"match_method": mm_match[i]})))
            edges_new.append((benchmark_lid, nid, "ligand_also_in_chembl", "{}"))

    if not prov.is_empty():
        # Same defensive extraction for the 7M+ provenance rows.
        prov_clean = prov.filter(pl.col("activity_id").is_not_null())
        p_aid = prov_clean["activity_id"].to_list()
        p_mol = prov_clean["molecule_chembl_id"].to_list()
        p_stype = prov_clean["standard_type"].to_list()
        p_sval = prov_clean["standard_value"].to_list()
        p_sun = prov_clean["standard_units"].to_list()
        p_pch = prov_clean["pchembl_value"].to_list()
        p_asy = prov_clean["assay_chembl_id"].to_list()
        p_doc = prov_clean["document_chembl_id"].to_list()
        p_tgt = prov_clean["target_chembl_id"].to_list()

        assays_seen, docs_seen, targets_seen, acts_seen = set(), set(), set(), set()
        for i in range(prov_clean.height):
            aid_raw = p_aid[i]
            if aid_raw is None:
                continue
            aid = int(aid_raw)
            if aid in acts_seen:
                continue
            acts_seen.add(aid)
            mol_chid = p_mol[i]
            chembl_lid = f"chembl_lig:{mol_chid}"
            act_nid = f"chembl_act:{aid}"
            nodes_new.append((act_nid, "ChEMBLActivity", str(aid),
                              json.dumps({"standard_type": p_stype[i],
                                           "standard_value": p_sval[i],
                                           "standard_units": p_sun[i],
                                           "pchembl_value": p_pch[i]})))
            edges_new.append((act_nid, chembl_lid, "chembl_activity_has_ligand", "{}"))
            asy_chid = p_asy[i]
            if asy_chid:
                asy_nid = f"chembl_asy:{asy_chid}"
                if asy_chid not in assays_seen:
                    nodes_new.append((asy_nid, "ChEMBLAssay", asy_chid, "{}"))
                    assays_seen.add(asy_chid)
                edges_new.append((act_nid, asy_nid, "chembl_activity_has_assay", "{}"))
                doc_chid = p_doc[i]
                if doc_chid:
                    edges_new.append((asy_nid, f"chembl_doc:{doc_chid}",
                                       "chembl_assay_from_document", "{}"))
            doc_chid = p_doc[i]
            if doc_chid:
                doc_nid = f"chembl_doc:{doc_chid}"
                if doc_chid not in docs_seen:
                    nodes_new.append((doc_nid, "ChEMBLDocument", doc_chid, "{}"))
                    docs_seen.add(doc_chid)
                edges_new.append((act_nid, doc_nid, "chembl_activity_has_document", "{}"))
                edges_new.append((doc_nid, "src:ChEMBL35", "chembl_document_from_source", "{}"))
            tgt_chid = p_tgt[i]
            if tgt_chid:
                tgt_nid = f"chembl_tgt:{tgt_chid}"
                if tgt_chid not in targets_seen:
                    nodes_new.append((tgt_nid, "ChEMBLTarget", tgt_chid, "{}"))
                    targets_seen.add(tgt_chid)
                edges_new.append((act_nid, tgt_nid, "chembl_activity_has_target", "{}"))

    # ---- BindingDB ligand + record + publication + target subgraph ----
    # (Enriched: BindingDB records bring Publication (pmid/doi) and Protein
    # (UniProt) provenance into the KG. This expands the audit "did model X
    # train on this protein?" signal beyond the ChEMBL-only path.)
    mp_bdb = pl.read_parquet(PROCESSED / "benchmark_to_bindingdb_ligand_map.parquet")
    # Same defect as the ChEMBL side above, same shape. The edge emitted below is
    # (benchmark Ligand -> bdb_lig:<inchikey>), and the benchmark Ligand id is
    # md5(canonical_smiles) — the dataset does not appear in it. Deduping on
    # (inchikey, benchmark_dataset) therefore keeps ONE canonical_smiles per
    # (compound, corpus) and drops the BindingDB link for every other SMILES of the
    # same compound. Key the dedup on what the edge is actually keyed on.
    mapped_bdb = (mp_bdb.filter(pl.col("match_method") != "unmatched")
                        .unique(subset=["canonical_smiles", "inchikey"]))
    bdb_lig = pl.read_parquet(PROCESSED / "bindingdb_ligands_minimal.parquet")
    bdb_lig_ik = bdb_lig.unique(subset=["ligand_inchikey"])
    mapped_with_bdb = mapped_bdb.join(bdb_lig_ik.rename({"ligand_inchikey": "inchikey"}),
                                       on="inchikey", how="left")
    b_ik = mapped_with_bdb["inchikey"].to_list()
    b_smi = mapped_with_bdb["canonical_smiles"].to_list()
    b_lsmi = mapped_with_bdb["ligand_smiles"].to_list()
    b_chid = mapped_with_bdb["chembl_id_ligand"].to_list()
    b_zinc = mapped_with_bdb["zinc_id_ligand"].to_list()
    b_match = mapped_with_bdb["match_method"].to_list()
    seen_bdb_lig = set()
    benchmark_iks = set()
    for i in range(mapped_with_bdb.height):
        ik = b_ik[i]
        if not ik:
            continue
        benchmark_iks.add(ik)
        nid = f"bdb_lig:{ik}"
        if ik not in seen_bdb_lig:
            nodes_new.append((nid, "BindingDBLigand", ik,
                              json.dumps({"smiles": b_lsmi[i],
                                           "chembl_id_ligand": b_chid[i],
                                           "zinc_id_ligand": b_zinc[i]})))
            edges_new.append((nid, "src:BindingDB202605", "bindingdb_record_from_source", "{}"))
            seen_bdb_lig.add(ik)
        bench_smi = b_smi[i]
        if bench_smi:
            benchmark_lid = _lig_node_id(bench_smi)
            edges_new.append((benchmark_lid, nid, "ligand_also_in_bindingdb", "{}"))
            edges_new.append((benchmark_lid, nid,
                              "benchmark_ligand_same_inchikey_as_bindingdb_ligand",
                              json.dumps({"match_method": b_match[i]})))

    # Now bring in the BindingDB record-level provenance for the mapped ligands:
    # Publication (pmid/doi) and Protein (UniProt) nodes.
    bdb_rec_path = PROCESSED / "bindingdb_records_minimal.parquet"
    if bdb_rec_path.exists() and benchmark_iks:
        bdb_rec = pl.read_parquet(bdb_rec_path)
        # Restrict to records whose ligand has been mapped to a benchmark.
        bdb_rec_b = bdb_rec.filter(pl.col("ligand_inchikey").is_in(list(benchmark_iks)))
        log.info("BindingDB record enrichment: %d records for %d mapped ligands",
                 bdb_rec_b.height, len(benchmark_iks))
        # Extract columns to lists (avoid the iter_rows null-byte bug).
        r_ik = bdb_rec_b["ligand_inchikey"].to_list() if "ligand_inchikey" in bdb_rec_b.columns else []
        r_pmid = bdb_rec_b["pmid"].to_list() if "pmid" in bdb_rec_b.columns else [None] * bdb_rec_b.height
        r_doi = bdb_rec_b["article_doi"].to_list() if "article_doi" in bdb_rec_b.columns else [None] * bdb_rec_b.height
        r_uniprot = bdb_rec_b["uniprot_swissprot_id"].to_list() if "uniprot_swissprot_id" in bdb_rec_b.columns else [None] * bdb_rec_b.height
        r_record_id = bdb_rec_b["bindingdb_record_id"].to_list() if "bindingdb_record_id" in bdb_rec_b.columns else [None] * bdb_rec_b.height
        seen_pub: set = set()
        seen_prot: set = set()
        seen_assay: set = set()
        n_pub = n_prot = n_assay = 0
        for j in range(len(r_ik)):
            ik = r_ik[j]
            if not ik:
                continue
            bdb_lid = f"bdb_lig:{ik}"
            # Publication: prefer PMID; fall back to DOI.
            pmid = (r_pmid[j] or "").strip()
            doi = (r_doi[j] or "").strip()
            if pmid:
                pub_nid = f"pub:pmid:{pmid}"
                if pmid not in seen_pub:
                    nodes_new.append((pub_nid, "Publication", pmid,
                                      json.dumps({"pmid": pmid, "doi": doi or None})))
                    edges_new.append((pub_nid, "src:BindingDB202605", "publication_from_source", "{}"))
                    seen_pub.add(pmid)
                    n_pub += 1
                edges_new.append((bdb_lid, pub_nid, "bindingdb_ligand_in_publication", "{}"))
            elif doi:
                pub_nid = f"pub:doi:{doi}"
                if doi not in seen_pub:
                    nodes_new.append((pub_nid, "Publication", doi,
                                      json.dumps({"doi": doi})))
                    edges_new.append((pub_nid, "src:BindingDB202605", "publication_from_source", "{}"))
                    seen_pub.add(doi)
                    n_pub += 1
                edges_new.append((bdb_lid, pub_nid, "bindingdb_ligand_in_publication", "{}"))
            # Protein from UniProt SwissProt accession.
            uniprot = (r_uniprot[j] or "").strip()
            if uniprot:
                prot_nid = f"protein:{uniprot}"
                if uniprot not in seen_prot:
                    nodes_new.append((prot_nid, "Protein", uniprot,
                                      json.dumps({"uniprot": uniprot, "source": "BindingDB"})))
                    seen_prot.add(uniprot)
                    n_prot += 1
                edges_new.append((bdb_lid, prot_nid, "bindingdb_ligand_targets_protein", "{}"))
            # Assay-like node from BindingDB record id (one per measurement).
            rec_id = r_record_id[j]
            if rec_id is not None and str(rec_id).strip():
                rec_str = str(rec_id).strip()
                asy_nid = f"bdb_rec:{rec_str}"
                if rec_str not in seen_assay:
                    nodes_new.append((asy_nid, "Assay", rec_str,
                                      json.dumps({"source": "BindingDB",
                                                   "bindingdb_record_id": rec_str})))
                    seen_assay.add(rec_str)
                    n_assay += 1
                edges_new.append((asy_nid, bdb_lid, "bindingdb_record_has_ligand", "{}"))
                if uniprot:
                    edges_new.append((asy_nid, f"protein:{uniprot}", "bindingdb_record_has_protein", "{}"))
        log.info("BindingDB enrichment emitted: %d Publication, %d Protein, %d Assay nodes",
                 n_pub, n_prot, n_assay)

    # ---- Persist ----
    n_df = pl.DataFrame(nodes_new, schema=["node_id", "node_type", "label", "props"], orient="row")
    e_df = pl.DataFrame(edges_new, schema=["src", "dst", "edge_type", "props"], orient="row")
    nodes = pl.concat([base_n, n_df], how="vertical_relaxed").rechunk().unique(subset=["node_id"])

    # This used to be:
    #     nodes = nodes.filter(pl.col("node_id").str.contains(":"))
    # described as "belt-and-suspenders", on the belief that "the iter_rows null-byte
    # bug that poisoned ~4M nodes in the earlier build has been fixed". It is not
    # fixed. And the filter is not a safety net — it is the thing that HIDES the bug:
    # it silently deletes every corrupted node, the semi-join below then silently
    # deletes every edge that touched one, and the build reports success. That is
    # precisely how a KG with 48,207 `example_has_ligand` edges instead of 5,025,493
    # got written to disk without a single warning.
    #
    # A node_id without ':' is not a node we can drop, it is proof that the frame in
    # memory is corrupt. Say so, and die.
    _bad = nodes.filter(~pl.col("node_id").str.contains(":")).height
    if _bad:
        raise RuntimeError(
            f"CORRUPTION: {_bad:,} node_ids have no ':' prefix — the string buffers "
            f"were zeroed in memory (polars). The RUN is bad, not the code; re-run. "
            f"Do NOT filter these away: dropping them silently deletes their edges too.")

    edges = pl.concat([base_e, e_df], how="vertical_relaxed").rechunk().unique()
    _bade = edges.filter(pl.col("edge_type").is_null() | (pl.col("edge_type") == "")).height
    if _bade:
        raise RuntimeError(
            f"CORRUPTION: {_bade:,} edges have an empty edge_type — same bug, same "
            f"remedy: re-run. Filtering them away would hide it.")
    valid = nodes.select("node_id")
    edges = (edges.join(valid.rename({"node_id": "src"}), on="src", how="semi")
                  .join(valid.rename({"node_id": "dst"}), on="dst", how="semi"))
    # Build-time invariants — fail loudly so regressions are caught at write
    # time rather than later by audit. If any of these fire, look for new
    # iter_rows / unchecked f-string interpolation paths.
    _dup = nodes.group_by("node_id").len().filter(pl.col("len") > 1)
    if _dup.height:
        sample = _dup.sort("len", descending=True).head(10).to_dicts()
        # Also pull the conflicting rows for the first ID so we can see types.
        first_id = sample[0]["node_id"] if sample else None
        first_rows = (nodes.filter(pl.col("node_id") == first_id).head(5)
                      .to_dicts() if first_id else [])
        raise RuntimeError(
            f"INVARIANT FAIL: {_dup.height} duplicate node_id rows after dedup. "
            f"Top dup IDs: {sample}. First dup rows: {first_rows}")
    _null_byte = nodes.filter(~pl.col("node_id").str.contains(":"))
    if _null_byte.height:
        raise RuntimeError(
            f"INVARIANT FAIL: {_null_byte.height} node_ids missing ':' prefix")
    _ids = nodes.select("node_id")
    _dangle_src = edges.join(_ids.rename({"node_id": "src"}), on="src", how="anti").height
    _dangle_dst = edges.join(_ids.rename({"node_id": "dst"}), on="dst", how="anti").height
    if _dangle_src or _dangle_dst:
        raise RuntimeError(
            f"INVARIANT FAIL: dangling edges src={_dangle_src} dst={_dangle_dst}")
    log.info("build-time invariants passed: 0 dup, 0 null-byte, 0 dangling")

    # The three invariants above all look at `node_id`, and node_id is the ONE string
    # column the corruption spares. On the bad run of 2026-07-14 they passed cleanly
    # while `node_type` held 1,128,590 NUL-filled values and `example_has_ligand` had
    # collapsed from 5,025,493 edges to 48,207. They were blind in exactly the shape
    # of the bug. This scans every string column, and checks the counts that cannot
    # legitimately move.
    n_expected = sum(pl.read_parquet(PROCESSED / f"{slug}_examples.parquet").height
                     for _, slug in CORPORA
                     if (PROCESSED / f"{slug}_examples.parquet").exists())
    validate_raw_kg(nodes, edges, n_expected)
    log.info("raw-KG write-time validation passed (%d expected Examples)", n_expected)

    nodes.write_parquet(PROCESSED / "kg_nodes.parquet")
    edges.write_parquet(PROCESSED / "kg_edges.parquet")

    nbt = nodes.group_by("node_type").agg(pl.len().alias("n")).sort("node_type")
    eet = edges.group_by("edge_type").agg(pl.len().alias("n")).sort("edge_type")
    (REPORTS / "kg_build_summary.md").write_text(
        "# KG build summary\n\n" + ts() + "\n\n"
        f"Corpora loaded: {', '.join(loaded)}\n\n"
        f"Nodes: **{nodes.height:,}** | Edges: **{edges.height:,}**\n\n"
        "## Nodes by type\n\n"
        + "\n".join(f"- {r['node_type']}: {r['n']:,}" for r in nbt.iter_rows(named=True))
        + "\n\n## Edges by type\n\n"
        + "\n".join(f"- {r['edge_type']}: {r['n']:,}" for r in eet.iter_rows(named=True))
        + "\n", encoding="utf-8")
    return (f"nodes={nodes.height:,}, edges={edges.height:,}, "
            f"chembl_lig={len(mapped_mol):,}, bdb_lig={len(seen_bdb_lig):,}, "
            f"cross_src_inchikey={cross_src}, cross_parent={cross_parent}")


# -------- main --------
TASKS = [
    # ChEMBL / BindingDB raw extracts (cached after first run).
    ("load_chembl",       task_load_chembl),
    ("load_bindingdb",    task_load_bindingdb),
    # Per-corpus loaders that produce <corpus>_examples/_nodes/_edges parquets.
    # Must run BEFORE chembl_map/bindingdb_map so their ligands are included
    # in the benchmark <-> reference cross-ref maps.
    # ALL FIVE corpora belong here. The three benchmark corpora used to be
    # absent, which let the build succeed on 17 % of the graph — see the comment
    # above task_load_dude.
    ("load_dude",         task_load_dude),
    ("load_dekois",       task_load_dekois),
    ("load_litpcba",      task_load_litpcba),
    ("load_bigbind",      task_load_bigbind),
    ("load_bayesbind",    task_load_bayesbind),
    # Cross-reference maps + activity provenance (depend on all corpus parquets).
    ("chembl_map",        task_chembl_map),
    ("bindingdb_map",     task_bindingdb_map),
    ("chembl_provenance", task_chembl_provenance),
    # Final KG assembly: concat per-corpus + ChEMBL/BindingDB cross-ref layer.
    ("build_kg",          task_build_kg),
]


def main() -> int:
    log_disk("build_kg_start", "vs-leakkg v3")
    ok = 0
    fail = 0
    for name, fn in TASKS:
        if run_task(name, fn):
            ok += 1
        else:
            fail += 1
    log_disk("build_kg_end", f"vs-leakkg v3 ok={ok} fail={fail}")
    print()
    print(f"KG build complete. {ok}/{len(TASKS)} tasks OK.")
    print("Main outputs:")
    print(" - data/processed/kg_nodes.parquet")
    print(" - data/processed/kg_edges.parquet")
    print(" - outputs/reports/kg_build_summary.md")
    if fail:
        print(f"\n{fail} task(s) failed. See outputs/reports/kg_build_status.md")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
