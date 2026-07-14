"""BayesBind V1.5 per-target loader.

Layout (per target):
  BayesBindV1.5/{test|val}/<TARGET_NAME>/
    actives.csv     # rich metadata: lig_smiles, standard_type, pchembl_value, uniprot, pocket, ...
    actives.smi     # SMILES, one per line
    random.csv      # decoys with pocket metadata
    random.smi
    pocket.pdb
    rec.pdb         # receptor variants
    rec_hs.pdb
    rec_nofix.pdb
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl


def discover_targets(root: Path) -> List[tuple[str, str]]:
    """Returns list of (split, target_name). `split` is 'test' or 'val'."""
    base = root / "BayesBindV1.5"
    base = base if base.exists() else root
    out: List[tuple[str, str]] = []
    for split in ("test", "val"):
        sdir = base / split
        if not sdir.exists():
            continue
        for t in sorted(p for p in sdir.iterdir() if p.is_dir()):
            if (t / "actives.csv").exists() or (t / "random.csv").exists():
                out.append((split, t.name))
    return out


def load_target(root: Path, split: str, target: str) -> pl.DataFrame:
    base = root / "BayesBindV1.5"
    base = base if base.exists() else root
    tdir = base / split / target
    frames = []
    for fname, label, label_type in (
        ("actives.csv", 1, "active"),
        ("random.csv",  0, "random"),
    ):
        p = tdir / fname
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            df = pl.read_csv(p, infer_schema_length=2000, ignore_errors=True)
        except Exception:
            continue
        if df.height == 0:
            continue
        # Standardize column name for SMILES (actives.csv uses lig_smiles; random.csv uses lig_smiles too).
        rename: dict = {}
        for c in df.columns:
            if c.lower() == "lig_smiles":
                rename[c] = "smiles_input"
        if rename:
            df = df.rename(rename)
        if "smiles_input" not in df.columns:
            continue
        df = df.with_columns([
            pl.lit("BayesBind").alias("source"),
            pl.lit(target).alias("target"),
            pl.lit(label, dtype=pl.Int8).alias("label"),
            pl.lit(label_type).alias("label_type"),
            pl.lit(split).alias("split"),
            pl.lit(fname).alias("source_file"),
        ])
        # Carry uniprot + pocket if available; otherwise null.
        keep_cols = ["smiles_input", "source", "target", "label", "label_type", "split", "source_file"]
        for opt in ("uniprot", "pocket", "standard_type", "standard_value",
                    "standard_units", "pchembl_value",
                    "ex_rec_pdb", "lig_cluster", "rec_cluster",
                    "num_pocket_residues"):
            if opt in df.columns:
                keep_cols.append(opt)
        frames.append(df.select(keep_cols))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def load_all(root: Path) -> pl.DataFrame:
    targets = discover_targets(root)
    parts = [load_target(root, split, t) for split, t in targets]
    parts = [p for p in parts if p.height > 0]
    if not parts:
        return pl.DataFrame()
    return pl.concat(parts, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# KG-ready builder — mirrors load_bigbind.build()
# ---------------------------------------------------------------------------
import logging as _logging
from typing import Optional


def _featurize_batch(smiles: list[str], log: Optional[_logging.Logger] = None
                     ) -> tuple[list[Optional[str]], list[Optional[str]],
                                list[Optional[str]], list[bool]]:
    """Parallel featurize via vsleakkg.chem (order preserved + length checked)."""
    from vsleakkg import chem as vc
    feats = vc.featurize_batch_parallel(smiles, log=log)
    canon = [f.smiles_canonical for f in feats]
    iks = [f.inchikey for f in feats]
    scaf = [f.scaffold_smiles for f in feats]
    ok = [f.parse_ok for f in feats]
    return canon, iks, scaf, ok


def build(extracted_dir: Path,
          log: Optional[_logging.Logger] = None
          ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build (examples, nodes, edges) frames from BayesBind raw.

    Returns the same triple as load_bigbind.build() so build_kg can wire it
    into the same per-corpus pipeline. Uses the BayesBind `uniprot` column
    from each target's actives.csv as the Protein target node id, falling
    back to the target directory name when uniprot is missing.
    """
    from vsleakkg import build_graph as vb
    if log is None:
        log = _logging.getLogger("vsleakkg.load_bayesbind")
    df = load_all(extracted_dir)
    if df.is_empty():
        raise RuntimeError("BayesBind: no examples loaded")
    log.info("BayesBind: %d raw rows across %d targets",
             df.height, df.select("target").n_unique())

    # Featurize SMILES through the same RDKit pipeline as every other corpus.
    log.info("BayesBind: re-canonicalizing SMILES via RDKit ...")
    canon, iks, scaffolds, ok_list = _featurize_batch(df["smiles_input"].to_list(), log=log)
    df = df.with_columns([
        pl.Series("smiles_canonical", canon),
        pl.Series("inchikey", iks),
        pl.Series("scaffold_smiles", scaffolds),
        pl.Series("parse_ok", ok_list),
    ])
    bad = int((~df["parse_ok"]).sum())
    if bad:
        log.warning("BayesBind: %d rows failed RDKit parse (excluded)", bad)
    df = df.filter(pl.col("parse_ok"))

    # Map to the loader contract used by build_graph.build_examples_frame.
    # Use uniprot when available; fall back to the target directory name
    # (already in `target` column) so we never emit a null protein target.
    has_uniprot = "uniprot" in df.columns
    if has_uniprot:
        df = df.with_columns([
            pl.coalesce([pl.col("uniprot"), pl.col("target")]).alias("target"),
        ])
    df = df.with_columns([
        pl.lit("BayesBind").alias("source"),
        pl.col("source_file").alias("ext_id_1"),
        pl.lit(None, dtype=pl.Utf8).alias("ext_id_2"),
    ]).select([
        "smiles_canonical", "inchikey", "scaffold_smiles",
        "source", "target", "label", "label_type", "split",
        "ext_id_1", "ext_id_2",
    ])

    examples = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        examples,
        include_decoy_protocol=True,    # BayesBind random decoys are a protocol
        include_protein_target=True,
    )
    log.info("BayesBind: %d examples, %d nodes, %d edges",
             examples.height, nodes.height, edges.height)
    return examples, nodes, edges
