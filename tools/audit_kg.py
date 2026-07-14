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
    "protein_cluster_30":       ("Protein", "ProteinCluster"),
    "protein_cluster_50":       ("Protein", "ProteinCluster"),
    "protein_cluster_90":       ("Protein", "ProteinCluster"),
    # named source_*, but emitted example-level
    "source_decoy_protocol":    ("Example", "DecoyProtocol"),
    "decoy_protocol_in_class":  ("DecoyProtocol", "DecoyProtocolClass"),
}
PAIR_TYPES = ["ligand_exact", "ligand_parent_exact",
              "ligand_fingerprint_exact", "ligand_similar"]

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default="outputs/kg")
    a = ap.parse_args()
    d = Path(a.kg_dir)
    n = pl.scan_parquet(d / "canonical_nodes.parquet")
    e = pl.scan_parquet(d / "canonical_edges.parquet")
    print(f"KG: {n.select(pl.len()).collect().item():,} nodes / "
          f"{e.select(pl.len()).collect().item():,} edges")
    structural(n, e)
    semantic(n, e)
    print(f"\n{len(_fails)} failed check(s)")
    for f in _fails:
        print("  -", f)
    return len(_fails)


if __name__ == "__main__":
    sys.exit(main())
