"""(B) first cut: a GLOBAL cold-target split, then audited with the SAME leakage metric
used on the published splits. Hard constraint: same protein 30%-cluster -> same partition
(stricter than BayesBind, and closes BigBind's 32% distant-homology hole). Greedy balanced,
label-aware assignment. Then report within-split leakage per axis + partition/label balance.

Not the final optimizer (no soft ligand/provenance minimisation yet) — a baseline to see how
far the hard target constraint alone already carries us. Run:
PYTHONNOUSERSITE=1 POLARS_MAX_THREADS=1 PYTHONPATH=src python research/split_v1_coldtarget.py
"""
from __future__ import annotations
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


def group_labels(edge_types, tmin=None, cap=None):
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
    return examples.with_columns(pl.Series("group_id", lab[ex_i]))


PROTEIN = ["example_has_protein", "protein_exact", "protein_cluster_30", "protein_cluster_50", "protein_cluster_90"]
print("building cold-target groups (protein 30% families)...")
g = group_labels(PROTEIN).join(lab_y, on="ex")
grp = (g.group_by("group_id").agg(pl.len().alias("n"), pl.col("y").sum().alias("na"))
       .sort(["n", "group_id"], descending=[True, False]))
print(f"  {grp.height} target-family groups; largest {grp['n'][0]:,} = {100*grp['n'][0]/N:.1f}%")

# ---- greedy balanced, label-aware assignment ----
RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
tgt_n = {k: v * N for k, v in RATIO.items()}
tgt_a = {k: v * int(lab_y["y"].sum()) for k, v in RATIO.items()}
cur_n = {k: 0.0 for k in RATIO}
cur_a = {k: 0.0 for k in RATIO}
assign = {}
for gid, n, na in grp.iter_rows():
    # pick partition with the most remaining room, blending size + active deficit
    best, bestscore = None, -1e18
    for k in RATIO:
        room_n = (tgt_n[k] - cur_n[k]) / tgt_n[k]
        room_a = (tgt_a[k] - cur_a[k]) / tgt_a[k] if tgt_a[k] else 0.0
        score = room_n + 0.5 * room_a
        if score > bestscore:
            best, bestscore = k, score
    assign[gid] = best
    cur_n[best] += n
    cur_a[best] += na
amap = pl.DataFrame({"group_id": list(assign.keys()), "part": list(assign.values())})
split = g.join(amap, on="group_id").select("ex", "part", "y")
print("\npartition sizes / active fraction:")
print(split.group_by("part").agg(pl.len().alias("n"), (100 * pl.col("y").mean()).alias("active_pct")).sort("part"))

# ---- audit OUR split with the same leakage metric (eval=test, seen=train+val) ----
seen, evalp = {"train", "val"}, "test"
ex_part = dict(zip(split["ex"], split["part"]))


def our_leak(edge_types, tmin=None, cap=None):
    lab = group_labels(edge_types, tmin, cap)
    d = lab.join(split, on="ex")
    sg = set(d.filter(pl.col("part").is_in(list(seen)))["group_id"].to_list())
    ev = d.filter(pl.col("part") == evalp)
    res = {}
    for yv, nm in ((None, "all"), (1, "act"), (0, "ina")):
        e = ev if yv is None else ev.filter(pl.col("y") == yv)
        res[nm] = 100 * e.filter(pl.col("group_id").is_in(sg)).height / e.height if e.height else float("nan")
    return res


print("\nWITHIN-SPLIT leakage of OUR cold-target split (% of test in same group as train/val):")
print(f"  {'regime':<18} {'all':>7} {'act':>7} {'ina':>7}")
for name, ets, tmin, cap in [
    ("protein-clust30", ["example_has_protein", "protein_exact", "protein_cluster_30"], None, None),
    ("ligand-identity", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact"], None, None),
    ("ligand+sim0.80", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar"], 0.80, None),
    ("scaffold(cap1k)", ["example_has_ligand", "ligand_scaffold"], None, 1000),
    ("assay(cap1k)", ["example_from_assay", "example_from_publication"], None, 1000),
    ("HIGH-ORDER all", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar", "ligand_scaffold", "example_has_protein", "protein_exact", "protein_cluster_90"], 0.80, 1000),
]:
    r = our_leak(ets, tmin, cap)
    print(f"  {name:<18} {r['all']:6.1f}% {r['act']:6.1f}% {r['ina']:6.1f}%")
