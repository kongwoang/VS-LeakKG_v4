"""Invariant audit for the canonical KG. Exit code = number of failed checks.

Two tiers:
  STRUCTURAL — the graph is a well-formed graph.
  SEMANTIC   — the graph means what biology and the proposal say it means. These are
               the checks that catch defects producing *wrong* leakage numbers rather
               than a crash.

Run after ANY change to the KG.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from vsleakkg.kg import schema

# The schema declares no endpoint signatures, so this table IS the contract.
ENDPOINTS = {
    "example_has_ligand":       ("Example", "Ligand"),
    "example_has_protein":      ("Example", "Protein"),
    "example_from_source":      ("Example", "DatasetSource"),
    "example_from_assay":       ("Example", "Assay"),
    "example_from_publication": ("Example", "Publication"),
    "ligand_scaffold":          ("Ligand", "Scaffold"),
    "ligand_exact":             ("Ligand", "Ligand"),
    "ligand_parent_exact":      ("Ligand", "Ligand"),
    "ligand_fingerprint_exact": ("Ligand", "Ligand"),
    "ligand_similar":           ("Ligand", "Ligand"),
    "ligand_measured_protein":  ("Ligand", "Protein"),
    "protein_exact":            ("Protein", "Protein"),
    "protein_cluster_30":       ("Protein", "ProteinCluster"),
    "protein_cluster_50":       ("Protein", "ProteinCluster"),
    "protein_cluster_90":       ("Protein", "ProteinCluster"),
    # named source_*, but emitted example-level
    "source_decoy_protocol":    ("Example", "DecoyProtocol"),
    "decoy_protocol_in_class":  ("DecoyProtocol", "DecoyProtocolClass"),
    # Facts about the example, in no axis (schema.NON_AXIS_EDGE_TYPES).
    "example_in_split":         ("Example", "Split"),
    "example_has_label_type":   ("Example", "LabelType"),
}
PAIR_TYPES = ["ligand_exact", "ligand_parent_exact",
              "ligand_fingerprint_exact", "ligand_similar", "protein_exact"]

_fails: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def structural(n: pl.LazyFrame, e: pl.LazyFrame) -> None:
    print("\n=== STRUCTURAL ===")
    dup = n.group_by("node_id").len().filter(pl.col("len") > 1).select(pl.len()).collect().item()
    check(dup == 0, "no duplicate node_id", f"{dup:,}")

    bad = n.filter(~pl.col("node_id").str.contains(":")).select(pl.len()).collect().item()
    check(bad == 0, "every node_id carries its ':' type prefix", f"{bad:,}")

    loops = e.filter(pl.col("src") == pl.col("dst")).select(pl.len()).collect().item()
    check(loops == 0, "no self-loops", f"{loops:,}")

    ids = n.select("node_id")
    for side in ("src", "dst"):
        d = e.join(ids, left_on=side, right_on="node_id", how="anti").select(pl.len()).collect().item()
        check(d == 0, f"no dangling edges ({side})", f"{d:,}")

    # per-type: a global unique() on 75M edges is a memory hazard
    etypes = sorted(e.select("edge_type").unique().collect()["edge_type"].to_list())
    dup_e = sum((e.filter(pl.col("edge_type") == t).group_by(["src", "dst"])
                  .agg(pl.len().alias("k")).filter(pl.col("k") > 1)
                  .select(pl.len()).collect().item()) for t in etypes)
    check(dup_e == 0, "no duplicate (src, dst, edge_type)", f"{dup_e:,}")

    endpoints = pl.concat([e.select(pl.col("src").alias("id")),
                           e.select(pl.col("dst").alias("id"))]).unique()
    orph = n.join(endpoints, left_on="node_id", right_on="id", how="anti").collect()
    bad = orph.filter(~pl.col("node_type").is_in(["DatasetSource", "DecoyProtocol"])).height
    check(bad == 0, "no orphans (DatasetSource/DecoyProtocol are pinned)", f"{bad:,}")

    cols = n.collect_schema().names()
    check("degree" in cols and "is_hub" not in cols,
          "nodes record `degree` (a fact), not `is_hub` (a policy)", f"cols={cols}")


def semantic(n: pl.LazyFrame, e: pl.LazyFrame) -> None:
    print("\n=== SEMANTIC ===")
    types = n.select("node_id", "node_type")

    bad = 0
    for t, (st, dt) in ENDPOINTS.items():
        j = (e.filter(pl.col("edge_type") == t)
              .join(types, left_on="src", right_on="node_id").rename({"node_type": "s"})
              .join(types, left_on="dst", right_on="node_id").rename({"node_type": "d"}))
        bad += j.filter((pl.col("s") != st) | (pl.col("d") != dt)).select(pl.len()).collect().item()
    check(bad == 0, "every edge connects the declared node types", f"{bad:,}")

    etypes = set(e.select("edge_type").unique().collect()["edge_type"].to_list())
    missing = sorted(etypes - set(schema.DEFAULT_WEIGHTS))
    check(not missing, "every edge type has a leakage weight", f"missing: {missing}")

    ex = n.filter(pl.col("node_type") == "Example").select("node_id")
    for t, what in [("example_has_ligand", "ligand"), ("example_from_source", "source"),
                    ("example_has_protein", "target protein")]:
        cnt = e.filter(pl.col("edge_type") == t).group_by("src").agg(pl.len().alias("k"))
        j = ex.join(cnt, left_on="node_id", right_on="src", how="left").collect()
        miss, multi = j.filter(pl.col("k").is_null()).height, j.filter(pl.col("k") > 1).height
        note = (" — several proteins per Example bridge the whole graph and collapse "
                "the protein axis") if t == "example_has_protein" else ""
        check(miss == 0 and multi == 0, f"every Example has exactly one {what}",
              f"missing={miss:,} multiple={multi:,}{note}")

    lig = n.filter(pl.col("node_type") == "Ligand").select("node_id")
    sc = e.filter(pl.col("edge_type") == "ligand_scaffold").group_by("src").agg(pl.len().alias("k"))
    multi = (lig.join(sc, left_on="node_id", right_on="src", how="inner")
                .filter(pl.col("k") > 1).select(pl.len()).collect().item())
    check(multi == 0, "every Ligand has at most one scaffold", f"{multi:,} with >1")

    for t in PAIR_TYPES:
        uns = e.filter((pl.col("edge_type") == t) & (pl.col("src") >= pl.col("dst"))) \
               .select(pl.len()).collect().item()
        check(uns == 0, f"{t}: pairs stored sorted (src < dst)", f"{uns:,} unsorted")

    ehp = e.filter(pl.col("edge_type") == "example_has_protein").select(
        pl.col("src").alias("ex"), pl.col("dst").alias("prot"))
    inclu = e.filter(pl.col("edge_type").str.starts_with("protein_cluster_")).select("src").unique()
    total = ehp.select("ex").unique().select(pl.len()).collect().item()
    cov = (ehp.join(inclu, left_on="prot", right_on="src", how="semi")
              .select("ex").unique().select(pl.len()).collect().item())
    check(total == cov, "every Example is visible to the protein-family axis",
          f"{total - cov:,}/{total:,} reach no clustered protein")

    scn = n.filter(pl.col("node_type") == "Scaffold").select("label").collect()
    flat = scn.with_columns(pl.col("label").str.replace_all(r"[/\\]", "")
                              .str.replace_all("@", "").alias("f"))
    g = flat.group_by("f").agg(pl.len().alias("k")).filter(pl.col("k") > 1)
    excess = int((g["k"] - 1).sum()) if g.height else 0
    check(excess == 0, "no Scaffold node duplicated modulo stereochemistry",
          f"{excess:,} excess nodes")

    # decoy protocol: only inactives, and the class tier is what spans corpora
    lab = (n.filter(pl.col("node_type") == "Example")
            .select(pl.col("node_id").alias("ex"),
                    pl.col("props").str.json_path_match("$.label").alias("y")))
    sdp = e.filter(pl.col("edge_type") == "source_decoy_protocol").select(
        pl.col("src").alias("ex"), pl.col("dst").alias("proto"))
    act = sdp.join(lab, on="ex").filter(pl.col("y") == "1").select(pl.len()).collect().item()
    check(act == 0, "decoy protocols link only inactives",
          f"{act:,} ACTIVE examples linked to a decoy-generation protocol")

    cls = e.filter(pl.col("edge_type") == "decoy_protocol_in_class").select(
        pl.col("src").alias("proto"), pl.col("dst").alias("cls"))
    src = e.filter(pl.col("edge_type") == "example_from_source").select(
        pl.col("src").alias("ex"), pl.col("dst").alias("corpus"))
    span = (sdp.join(src, on="ex").join(cls, on="proto")
               .group_by("cls").agg(pl.col("corpus").n_unique().alias("nc")).collect())
    crossing = span.filter(pl.col("nc") > 1).height
    check(crossing > 0, "the decoy-protocol CLASS tier spans corpora",
          "no method class is shared by two corpora — the axis would just relabel source")


LIGAND_SIMILAR_MIN = 0.80
FINGERPRINT_EXACT_T = 0.9995


def facts(n: pl.LazyFrame, e: pl.LazyFrame, *, sample: int) -> None:
    """Do the values the KG RECORDS agree with the graph it SHIPS?

    The structural and semantic tiers check that the graph is well-formed and that
    its edges mean what they say. Neither of them ever compared a recorded number
    against the thing it summarises — so every one of these passed on a KG that had
    306,704 wrong degrees, 41.3 % wrong heavy-atom counts, a similarity threshold
    0.05 above its declared contract, and an identity relation that was a verbatim
    copy of another one. `check("degree" in cols)` is not a check on degree.
    """
    print("\n=== FACTS (recorded value vs the data it summarises) ===")

    # 1. degree. The fact that replaced the is_hub policy, and what the downstream
    #    weight-discount reads. Was computed BEFORE the final dedup passes.
    deg = (pl.concat([e.select(pl.col("src").alias("node_id")),
                      e.select(pl.col("dst").alias("node_id"))])
             .group_by("node_id").agg(pl.len().alias("real")))
    j = (n.select("node_id", "node_type", "degree")
          .join(deg, on="node_id", how="left")
          .with_columns(pl.col("real").fill_null(0))
          .filter(pl.col("degree") != pl.col("real")).collect())
    detail = f"{j.height:,} nodes disagree"
    if j.height:
        by = dict(j.group_by("node_type").agg(pl.len().alias("k")).iter_rows())
        detail += f" ({by}); worst delta={int((j['degree'] - j['real']).abs().max())}"
    check(j.height == 0, "`degree` equals the node's real edge count", detail)

    # 2. An edge type absent from ENDPOINTS is silently exempt from the endpoint
    #    check — the contract has to be total, or it is not a contract.
    etypes = set(e.select("edge_type").unique().collect()["edge_type"].to_list())
    unlisted = sorted(etypes - set(ENDPOINTS))
    check(not unlisted, "every edge type present has a declared endpoint signature",
          f"unlisted: {unlisted}")

    # 3. ligand_similar: one threshold, one provenance. The shipped graph mixed a
    #    0.85 global pass with 427 LIT-PCBA-only rows at 0.80–0.849.
    ls = (e.filter(pl.col("edge_type") == "ligand_similar")
           .select(pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64).alias("t"),
                   pl.col("props").str.json_path_match("$.method").alias("m"))
           .collect())
    if ls.height:
        n_null = ls.filter(pl.col("t").is_null()).height
        check(n_null == 0, "every ligand_similar edge records its Tanimoto", f"{n_null:,} without")
        lo, hi = ls["t"].min(), ls["t"].max()
        check(lo is not None and lo >= LIGAND_SIMILAR_MIN and hi < FINGERPRINT_EXACT_T,
              f"ligand_similar Tanimoto within the declared [{LIGAND_SIMILAR_MIN}, "
              f"{FINGERPRINT_EXACT_T})", f"observed [{lo}, {hi}]")
        methods = sorted(set(ls["m"].drop_nulls().to_list()))
        n_nom = ls.filter(pl.col("m").is_null()).height
        check(len(methods) <= 1 and n_nom == 0,
              "ligand_similar comes from ONE pass (one threshold, one method)",
              f"methods={methods}, {n_nom:,} edges with no method — a second, lower "
              f"threshold hiding inside the edge type")

    fe = (e.filter(pl.col("edge_type") == "ligand_fingerprint_exact")
           .select(pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64).alias("t"))
           .collect())
    if fe.height:
        below = fe.filter(pl.col("t") < FINGERPRINT_EXACT_T).height
        check(below == 0, f"ligand_fingerprint_exact Tanimoto >= {FINGERPRINT_EXACT_T}",
              f"{below:,} below")

    # 4. ligand_parent_exact must say something ligand_exact does not. It used to be
    #    a byte-identical copy of it (6,939 == 6,939, same pairs).
    le = e.filter(pl.col("edge_type") == "ligand_exact").select("src", "dst")
    lp = e.filter(pl.col("edge_type") == "ligand_parent_exact").select("src", "dst")
    overlap = le.join(lp, on=["src", "dst"], how="semi").select(pl.len()).collect().item()
    n_lp = lp.select(pl.len()).collect().item()
    check(overlap == 0,
          "ligand_parent_exact is disjoint from ligand_exact (different full InChIKey)",
          f"{overlap:,} of {n_lp:,} parent edges duplicate an exact edge")

    # 5. n_heavy_atoms — the fact the trivial-scaffold FILTER was replaced by. Used to
    #    count letters in the SMILES string: Cl/Br counted 2, the H in [nH] counted 1.
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        sc = (n.filter(pl.col("node_type") == "Scaffold")
               .select("label", pl.col("props").str.json_path_match("$.n_heavy_atoms")
                       .cast(pl.Int64).alias("rec")).collect())
        s = sc if sc.height <= sample else sc.sample(n=sample, seed=7)
        wrong = 0
        for lbl, rec in s.iter_rows():
            m = Chem.MolFromSmiles(lbl) if lbl else None
            true = m.GetNumHeavyAtoms() if m is not None else -1
            if rec != true:
                wrong += 1
        check(wrong == 0, "Scaffold props.n_heavy_atoms equals RDKit's heavy-atom count",
              f"{wrong:,}/{s.height:,} wrong ({100*wrong/max(1, s.height):.1f} %)")
    except ImportError:
        print("  [SKIP] n_heavy_atoms — RDKit unavailable")

    # 6. "at most one scaffold" is not the invariant; "exactly one" is. A Ligand with
    #    no scaffold is invisible to the scaffold axis and nothing said so.
    lig = n.filter(pl.col("node_type") == "Ligand").select("node_id")
    has = e.filter(pl.col("edge_type") == "ligand_scaffold").select("src").unique()
    miss = lig.join(has, left_on="node_id", right_on="src", how="anti").select(pl.len()).collect().item()
    check(miss == 0, "every Ligand has exactly one scaffold", f"{miss:,} with none")

    # 7. The class label is binary. DEKOIS's loader can emit -1 ("unknown") and every
    #    downstream tally silently compares only against "1" and "0".
    y = (n.filter(pl.col("node_type") == "Example")
          .select(pl.col("props").str.json_path_match("$.label").alias("y"))
          .group_by("y").agg(pl.len().alias("k")).collect())
    vals = sorted(y["y"].drop_nulls().to_list())
    check(vals == ["0", "1"], "Example label is exactly {0, 1}",
          f"observed {dict(y.iter_rows())}")

    # 8. A target edge is the corpus loader's claim. It used to be deduped against a
    #    BindingDB-derived edge, and the BindingDB row won: 305,668 target edges
    #    claimed props.source = "BindingDB", which is simply not where they came from.
    bad = (e.filter((pl.col("edge_type") == "example_has_protein")
                    & pl.col("props").str.contains("BindingDB"))
            .select(pl.len()).collect().item())
    check(bad == 0, "example_has_protein carries no BindingDB provenance",
          f"{bad:,} target edges attribute themselves to BindingDB")

    # 9. An axis relation that the schema declares, weights and lists — and that the
    #    graph never emits — is a hole nothing else can see. `protein_exact` sat at
    #    weight 1.00 inside AXIS_EDGE_TYPES["protein"] with ZERO edges, so two Protein
    #    nodes holding the same protein under different accessions were never joined.
    #
    #    This loop used to cover the ligand and protein axes only, which is how the
    #    TIME axis stayed empty without ever failing an audit: TimeBin,
    #    example_has_timebin and time_overlap are declared, weighted (1.00 and 0.40)
    #    and listed in AXES, and the graph has never contained one of them. Every
    #    axis is checked now. An axis with no edges is not an axis.
    for axis in sorted(schema.AXIS_EDGE_TYPES):
        for et in sorted(set(schema.AXIS_EDGE_TYPES[axis])):
            k = e.filter(pl.col("edge_type") == et).select(pl.len()).collect().item()
            check(k > 0, f"[{axis}] axis relation `{et}` is populated", f"{k:,} edges")

    # 10. protein_exact must NOT swallow the HIV domain split: the 99-aa protease is
    #     100 % identical to a slice of the 1,447-aa polyprotein, and merging them
    #     would claim a model that learned protease has learned reverse transcriptase.
    hiv = (e.filter((pl.col("edge_type") == "protein_exact")
                    & (pl.col("src").str.contains("HIV1:")
                       | pl.col("dst").str.contains("HIV1:"))
                    & (pl.col("props").str.json_path_match("$.alnlen")
                       .cast(pl.Int64) > 700))
            .select(pl.len()).collect().item())
    check(hiv == 0, "protein_exact does not merge an HIV domain into the polyprotein",
          f"{hiv:,} such edges")

    # 11. `label` is 0/1 and cannot tell the three kinds of negative apart. A DUD-E
    #     property-matched decoy, a LIT-PCBA measured inactive and a BayesBind random
    #     molecule are all label 0, and consolidate used to drop the one relation that
    #     separated them ("LabelType — static lookup table"). Without it the decoy-bias
    #     question is not even expressible. Every Example must carry a label type.
    ex = n.filter(pl.col("node_type") == "Example")
    n_ex = ex.select(pl.len()).collect().item()
    for field, expect in (("label_type", {"active", "decoy", "inactive", "random"}),
                          ("split", None)):
        got = (ex.select(pl.col("props").str.json_path_match(f"$.{field}").alias("v"))
                 .collect())
        n_null = int(got["v"].null_count())
        check(n_null == 0, f"every Example carries `{field}` in its props",
              f"{n_null:,} of {n_ex:,} Examples have no {field}")
        if expect is not None and n_null == 0:
            vals = set(got["v"].unique().to_list())
            check(vals == expect, "label_type has exactly the four corpus values",
                  f"got {sorted(vals)}, expected {sorted(expect)}")

    # 12. ...and the fix for #11 must not become a leak of its own. `lt:decoy` has
    #     degree 1.5 M: an axis that traversed it would join every decoy to every
    #     other decoy at weight 1.00 and score the corpus as totally contaminated.
    #     These edge types belong in the graph and in no axis.
    in_axis = {et for types in schema.AXIS_EDGE_TYPES.values() for et in types}
    leaked = sorted(schema.NON_AXIS_EDGE_TYPES & in_axis)
    check(not leaked, "no non-axis edge type has been added to an axis",
          f"{leaked} would connect every example sharing the attribute")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default="outputs/kg")
    ap.add_argument("--sample", type=int, default=50_000,
                    help="scaffolds to re-check with RDKit (0 = all)")
    a = ap.parse_args()
    d = Path(a.kg_dir)
    n = pl.scan_parquet(d / "canonical_nodes.parquet")
    e = pl.scan_parquet(d / "canonical_edges.parquet")
    print(f"KG: {n.select(pl.len()).collect().item():,} nodes / "
          f"{e.select(pl.len()).collect().item():,} edges")
    structural(n, e)
    semantic(n, e)
    facts(n, e, sample=a.sample or 10**9)
    print(f"\n{len(_fails)} failed check(s)")
    for f in _fails:
        print("  -", f)
    return len(_fails)


if __name__ == "__main__":
    sys.exit(main())
