"""(B2) 2D block split = cold-BOTH via typed KG groups. Assign each target-family and each
ligand-group to a partition; keep example (t,l) only if part(t)==part(l) (the diagonal),
drop the off-diagonal. The retained split is cold on ligand AND protein simultaneously — the
thing single-axis splits and v1 could not do. Cost = dropped fraction; we minimise it by
coordinate descent (alternate target<->ligand majority reassignment) under size targets.

Then audit the retained split with the same leakage metric. Run:
PYTHONNOUSERSITE=1 POLARS_MAX_THREADS=1 PYTHONPATH=src python research/block_split.py
"""
from __future__ import annotations
import sys
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

nodes = pl.scan_parquet("outputs/kg/canonical_nodes.parquet")
edges = pl.scan_parquet("outputs/kg/canonical_edges.parquet")
examples = nodes.filter(pl.col("node_type") == "Example").select(pl.col("node_id").alias("ex")).collect()
N = examples.height
DEG = dict(zip(nodes.select("node_id").collect()["node_id"], nodes.select("degree").collect()["degree"]))
lab_y = (nodes.filter(pl.col("node_type") == "Example")
         .select(pl.col("node_id").alias("ex"), pl.col("props").str.json_path_match("$.label").cast(pl.Int8).alias("y")).collect())
PARTS = ["train", "val", "test"]
RATIO = np.array([0.70, 0.15, 0.15])


def group_labels(edge_types, tmin=None, cap=None, name="grp"):
    sub = edges.filter(pl.col("edge_type").is_in(edge_types))
    if tmin is not None:
        sub = sub.filter((pl.col("edge_type") != "ligand_similar")
                         | (pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64) >= tmin))
    pairs = sub.select("src", "dst").collect()
    if cap is not None:
        hub = {v for v in set(pairs["src"].to_list()) | set(pairs["dst"].to_list())
               if not v.startswith("ex:") and DEG.get(v, 0) > cap}
        if hub:
            pairs = pairs.filter(~pl.col("src").is_in(hub) & ~pl.col("dst").is_in(hub))
    universe = pl.concat([examples["ex"], pairs["src"], pairs["dst"]]).unique().sort()
    idx = pl.DataFrame({"id": universe}).with_row_index("i")
    e2 = (pairs.join(idx.rename({"id": "src", "i": "si"}), on="src").join(idx.rename({"id": "dst", "i": "di"}), on="dst"))
    si, di = e2["si"].to_numpy(), e2["di"].to_numpy()
    nn = idx.height
    if len(si):
        _, lab = connected_components(coo_matrix((np.ones(len(si), np.int8), (si, di)), shape=(nn, nn)), directed=False)
    else:
        lab = np.arange(nn)
    ex_i = examples.join(idx, left_on="ex", right_on="id")["i"].to_numpy()
    return examples.with_columns(pl.Series(name, lab[ex_i].astype(np.int64)))


TARGET = ["example_has_protein", "protein_exact", "protein_cluster_30", "protein_cluster_50", "protein_cluster_90"]
LIGAND = ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar"]
print("computing target/ligand blocks ...")
base = (examples.join(group_labels(TARGET, name="tb"), on="ex")
        .join(group_labels(LIGAND, 0.80, name="lb"), on="ex").join(lab_y, on="ex"))
tb = base["tb"].to_numpy(); lb = base["lb"].to_numpy(); y = base["y"].to_numpy()
n_tb, n_lb = tb.max() + 1, lb.max() + 1
print(f"  {n_tb:,} target-blocks, {n_lb:,} ligand-blocks, {N:,} examples")


def sizes_by(block, nblk):
    s = np.zeros(nblk, np.int64); np.add.at(s, block, 1); return s


tb_size = sizes_by(tb, n_tb)


def assign_targets_by_size():
    """size-greedy: largest target-blocks fill partitions toward RATIO*N."""
    part = np.full(n_tb, -1, np.int64)
    cur = np.zeros(3); cap = RATIO * N
    for g in np.argsort(-tb_size):
        p = int(np.argmax((cap - cur) / cap))
        part[g] = p; cur[p] += tb_size[g]
    return part


def majority_reassign(block, nblk, other_part_of_example):
    """assign each block to the partition holding most of its examples (by other side)."""
    w = np.zeros((nblk, 3), np.int64)
    np.add.at(w, (block, other_part_of_example), 1)
    return w.argmax(1)


# Targets are the sizing anchor: fix them by size-greedy (guarantees ~70/15/15), then let
# ligand-blocks follow (majority of their examples' target-partition). Majority is optimal for
# the diagonal given fixed targets; letting targets drift by majority collapses everything into
# one partition, so we don't.
tpart = assign_targets_by_size()
lpart = majority_reassign(lb, n_lb, tpart[tb])
diag = (tpart[tb] == lpart[lb])
print(f"  retained diagonal (fixed-target + ligand-majority) = {100*diag.mean():.1f}%")

epart = np.where(tpart[tb] == lpart[lb], tpart[tb], -1)   # -1 = dropped (off-diagonal)
clean = epart >= 0
split = base.with_columns(pl.Series("p", epart)).filter(pl.col("p") >= 0)
print(f"\nRETAINED (cold-both) = {clean.sum():,} / {N:,} = {100*clean.mean():.1f}%  (dropped off-diagonal {100*(1-clean.mean()):.1f}%)")
sz = split.group_by("p").agg(pl.len().alias("n"), (100*pl.col("y").mean()).alias("act%")).sort("p")
print(sz.with_columns(pl.col("p").replace_strict({0:"train",1:"val",2:"test"})))

# ---- audit the retained split (eval=test=2, seen=train+val={0,1}) ----
sp = dict(zip(split["ex"], split["p"]))
clean_ids = set(split["ex"].to_list())
print("\nWITHIN-SPLIT leakage of RETAINED block split (% of test in same group as train/val):")
print(f"  {'regime':<18} {'all':>7} {'act':>7} {'ina':>7}")
part_df = split.select("ex", "p", "y")
for name, ets, tmin, cap in [
    ("protein-30 (block)", ["example_has_protein", "protein_exact", "protein_cluster_30"], None, None),
    ("ligand+sim (block)", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar"], 0.80, None),
    ("scaffold(cap1k)", ["example_has_ligand", "ligand_scaffold"], None, 1000),
    ("assay(cap1k)", ["example_from_assay", "example_from_publication"], None, 1000),
    ("HIGH-ORDER all", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar", "ligand_scaffold", "example_has_protein", "protein_exact", "protein_cluster_90"], 0.80, 1000),
]:
    lab = group_labels(ets, tmin, cap)
    d = lab.join(part_df, on="ex")
    sg = set(d.filter(pl.col("p").is_in([0, 1]))["grp"].to_list())
    ev = d.filter(pl.col("p") == 2)
    r = {}
    for yv, nm in ((None, "all"), (1, "act"), (0, "ina")):
        e = ev if yv is None else ev.filter(pl.col("y") == yv)
        r[nm] = 100 * e.filter(pl.col("grp").is_in(sg)).height / e.height if e.height else float("nan")
    print(f"  {name:<18} {r['all']:6.1f}% {r['act']:6.1f}% {r['ina']:6.1f}%")
