"""(B2) v4 FINAL: cold-multi-axis (ligand+scaffold+protein) block split, label-aware target
assignment, saved as a usable artifact. Reports the residual head-to-head vs published splits.
Run: PYTHONNOUSERSITE=1 POLARS_MAX_THREADS=1 PYTHONPATH=src python research/block_split_v4.py
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
RATIO = np.array([0.70, 0.15, 0.15])


def group_labels(edge_types, tmin=None, cap=None, cap_prefix=None, name="grp"):
    sub = edges.filter(pl.col("edge_type").is_in(edge_types))
    if tmin is not None:
        sub = sub.filter((pl.col("edge_type") != "ligand_similar")
                         | (pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64) >= tmin))
    pairs = sub.select("src", "dst").collect()
    if cap is not None:
        alln = set(pairs["src"].to_list()) | set(pairs["dst"].to_list())
        def ishub(v):
            if v.startswith("ex:"): return False
            if cap_prefix and not v.startswith(cap_prefix): return False
            return DEG.get(v, 0) > cap
        hub = {v for v in alln if ishub(v)}
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
LIGAND_SC = ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar", "ligand_scaffold"]
print("computing blocks ...")
tb = group_labels(TARGET, name="tb")["tb"].to_numpy()
lb = group_labels(LIGAND_SC, 0.80, cap=1000, cap_prefix="sca:", name="lb")["lb"].to_numpy()
y = lab_y["y"].to_numpy()
n_tb, n_lb = tb.max() + 1, lb.max() + 1
tb_size = np.zeros(n_tb, np.int64); np.add.at(tb_size, tb, 1)
tb_act = np.zeros(n_tb, np.int64); np.add.at(tb_act, tb, y)
order = np.argsort(-tb_size)
A = int(y.sum())


def assign_targets(budget):
    """size-greedy toward sqrt(budget)*N, tie-broken to also balance active count."""
    part = np.zeros(n_tb, np.int64); cur = np.zeros(3); cura = np.zeros(3)
    capN, capA = budget * N, budget * A
    for g in order:
        room = (capN - cur) / capN + 0.5 * (capA - cura) / (capA + 1.0)
        p = int(np.argmax(room)); part[g] = p; cur[p] += tb_size[g]; cura[p] += tb_act[g]
    return part


def majority(block, nblk, other):
    w = np.zeros((nblk, 3), np.int64); np.add.at(w, (block, other), 1); return w.argmax(1)


budget = np.sqrt(RATIO); budget /= budget.sum()
tpart = assign_targets(budget)
lpart = majority(lb, n_lb, tpart[tb])
epart = np.where(tpart[tb] == lpart[lb], tpart[tb], -1)
split = examples.with_columns(pl.Series("p", epart), pl.Series("y", y)).filter(pl.col("p") >= 0)
kept = (epart >= 0).mean()
print(f"RETAINED (cold ligand+scaffold+protein) = {(epart>=0).sum():,}/{N:,} = {100*kept:.1f}%  (dropped {100*(1-kept):.1f}%)")
print(split.group_by("p").agg(pl.len().alias("n"), (100*pl.col("y").mean()).round(1).alias("act%")).sort("p")
      .with_columns(pl.col("p").replace_strict({0: "train", 1: "val", 2: "test"}).alias("part")))

# save artifact
out = (examples.with_columns(pl.Series("p", epart))
       .with_columns(pl.when(pl.col("p") == 0).then(pl.lit("train")).when(pl.col("p") == 1).then(pl.lit("val"))
                     .when(pl.col("p") == 2).then(pl.lit("test")).otherwise(pl.lit("dropped")).alias("partition"))
       .select(pl.col("ex").alias("example_id"), "partition"))
import os
os.makedirs("outputs/splits", exist_ok=True)
out.write_parquet("outputs/splits/vsleakkg_coldmulti_v4.parquet")
print(f"saved -> outputs/splits/vsleakkg_coldmulti_v4.parquet ({out.height:,} rows)")

print("\nWITHIN-SPLIT leakage of v4 (% of test active/inactive in same group as train/val):")
print(f"  {'regime':<20} {'all':>7} {'act':>7} {'ina':>7}")
pf = split.select("ex", "p", "y")
for name, ets, tmin, cap, cp in [
    ("protein-30", ["example_has_protein", "protein_exact", "protein_cluster_30"], None, None, None),
    ("ligand+sim", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar"], 0.80, None, None),
    ("scaffold", ["example_has_ligand", "ligand_scaffold"], None, 1000, "sca:"),
    ("assay", ["example_from_assay", "example_from_publication"], None, 1000, None),
    ("HIGH-ORDER all", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar", "ligand_scaffold", "example_has_protein", "protein_exact", "protein_cluster_90"], 0.80, 1000, None),
]:
    lab = group_labels(ets, tmin, cap, cp)
    d = lab.join(pf, on="ex")
    sg = set(d.filter(pl.col("p").is_in([0, 1]))["grp"].to_list())
    ev = d.filter(pl.col("p") == 2)
    r = {}
    for yv, nm in ((None, "all"), (1, "act"), (0, "ina")):
        e = ev if yv is None else ev.filter(pl.col("y") == yv)
        r[nm] = 100 * e.filter(pl.col("grp").is_in(sg)).height / e.height if e.height else float("nan")
    print(f"  {name:<20} {r['all']:6.1f}% {r['act']:6.1f}% {r['ina']:6.1f}%")
