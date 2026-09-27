"""Paper-ready hardening of the published-split leakage audit.
  (1) within-split leakage stratified by LABEL (kills the provenance/label confound)
  (2) pretraining leakage, GRADED: exact pair -> + similar ligand -> + homologous protein
      (the high-order 2-hop channel no split method measures)
  (3) which BigBind test targets carry the 32% distant-homology (30%) leak
Run: PYTHONNOUSERSITE=1 POLARS_MAX_THREADS=1 PYTHONPATH=src python enhanced_audit.py
"""
from __future__ import annotations
import time
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

nodes = pl.scan_parquet("outputs/kg/canonical_nodes.parquet")
edges = pl.scan_parquet("outputs/kg/canonical_edges.parquet")

examples = nodes.filter(pl.col("node_type") == "Example").select(pl.col("node_id").alias("ex")).collect()
DEG = dict(zip(nodes.select("node_id").collect()["node_id"],
               nodes.select("degree").collect()["degree"]))
lab_y = (nodes.filter(pl.col("node_type") == "Example")
         .select(pl.col("node_id").alias("ex"),
                 pl.col("props").str.json_path_match("$.label").alias("y")).collect())
eis = (edges.filter(pl.col("edge_type") == "example_in_split")
       .select(pl.col("src").alias("ex"), pl.col("dst").alias("split")).collect()
       .with_columns(pl.col("split").str.split(":").list.get(1).alias("corpus"),
                     pl.col("split").str.split(":").list.get(2).alias("part")))
SPLIT_ROLE = {"LIT-PCBA": ("validation", {"train"}),
              "BigBind": ("test", {"train", "val"}),
              "BayesBind": ("test", {"val"})}


def group_labels(edge_types, tmin, cap):
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
    e2 = (pairs.join(idx.rename({"id": "src", "i": "si"}), on="src")
                .join(idx.rename({"id": "dst", "i": "di"}), on="dst"))
    si, di = e2["si"].to_numpy(), e2["di"].to_numpy()
    nn = idx.height
    if len(si):
        _, lab = connected_components(coo_matrix((np.ones(len(si), np.int8), (si, di)), shape=(nn, nn)), directed=False)
    else:
        lab = np.arange(nn)
    ex_i = examples.join(idx, left_on="ex", right_on="id")["i"].to_numpy()
    return examples.with_columns(pl.Series("group_id", lab[ex_i]))


# ---------- (1) label-stratified within-split leakage ----------
print("=" * 96)
print("(1) WITHIN-SPLIT leakage stratified by label  (active% / inactive%)")
print("=" * 96)
REG = [
    ("ligand-identity", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact"], None, None),
    ("ligand+sim0.80", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar"], 0.80, None),
    ("protein-clust30", ["example_has_protein", "protein_exact", "protein_cluster_30"], None, None),
    ("HIGH-ORDER all", ["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact", "ligand_similar", "ligand_scaffold", "example_has_protein", "protein_exact", "protein_cluster_90"], 0.80, 1000),
]
print(f"{'regime':<18} {'LIT-PCBA a/i':>18} {'BigBind a/i':>18} {'BayesBind a/i':>18}")
print("-" * 96)
for name, ets, tmin, cap in REG:
    t0 = time.time()
    d = eis.join(group_labels(ets, tmin, cap), on="ex").join(lab_y, on="ex")
    cells = []
    for corpus, (evalp, seen) in SPLIT_ROLE.items():
        c = d.filter(pl.col("corpus") == corpus)
        sg = set(c.filter(pl.col("part").is_in(list(seen)))["group_id"].to_list())
        ev = c.filter(pl.col("part") == evalp)
        parts = []
        for yv in ("1", "0"):
            evy = ev.filter(pl.col("y") == yv)
            p = 100 * evy.filter(pl.col("group_id").is_in(sg)).height / evy.height if evy.height else float("nan")
            parts.append(p)
        cells.append(f"{parts[0]:5.1f}/{parts[1]:5.1f}")
    print(f"{name:<18} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}   [{time.time()-t0:.0f}s]")

# ---------- (2) graded pretraining leakage ----------
print("\n" + "=" * 96)
print("(2) PRETRAINING leakage of EVAL examples: is (ligand,protein) in ChEMBL/BindingDB, graded")
print("    exact -> + similar-ligand(T>=0.8) -> + homolog-protein(90%) -> either")
print("=" * 96)
M = (edges.filter(pl.col("edge_type") == "ligand_measured_protein")
     .select(pl.col("src").alias("lig"), pl.col("dst").alias("prot")).collect().unique())
NB = pl.concat([
    edges.filter(pl.col("edge_type").is_in(["ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact"]))
         .select(pl.col("src").alias("lig"), pl.col("dst").alias("lig2")).collect(),
    edges.filter(pl.col("edge_type") == "ligand_similar")
         .filter(pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64) >= 0.80)
         .select(pl.col("src").alias("lig"), pl.col("dst").alias("lig2")).collect(),
])
NB = pl.concat([NB, NB.select(pl.col("lig2").alias("lig"), pl.col("lig").alias("lig2"))]).unique()  # symmetric
CL90 = (edges.filter(pl.col("edge_type") == "protein_cluster_90")
        .select(pl.col("src").alias("prot"), pl.col("dst").alias("cl")).collect())
Mc = M.join(CL90, on="prot").select("lig", "cl").unique()  # (ligand, protein-cluster)

ehl = edges.filter(pl.col("edge_type") == "example_has_ligand").select(pl.col("src").alias("ex"), pl.col("dst").alias("lig")).collect()
ehp = edges.filter(pl.col("edge_type") == "example_has_protein").select(pl.col("src").alias("ex"), pl.col("dst").alias("prot")).collect()
xpair = ehl.join(ehp, on="ex")

print(f"{'corpus':<12} {'exact':>8} {'+simLig':>9} {'+homolProt':>11} {'either2hop':>11}")
for corpus, (evalp, seen) in SPLIT_ROLE.items():
    ev_ids = eis.filter((pl.col("corpus") == corpus) & (pl.col("part") == evalp)).select("ex")
    ev = ev_ids.join(xpair, on="ex")  # ex, lig, prot
    ntot = ev.height
    hit_exact = ev.join(M, on=["lig", "prot"]).select("ex").unique()
    ev_nb = ev.join(NB, on="lig").select("ex", pl.col("lig2").alias("lig"), "prot")
    hit_sim = pl.concat([ev.select("ex", "lig", "prot"), ev_nb]).join(M, on=["lig", "prot"]).select("ex").unique()
    ev_cl = ev.join(CL90, on="prot").select("ex", "lig", "cl")
    hit_hom = ev_cl.join(Mc, on=["lig", "cl"]).select("ex").unique()
    either = pl.concat([hit_sim, hit_hom]).unique()
    f = lambda h: 100 * h.height / ntot if ntot else float("nan")
    print(f"{corpus:<12} {f(hit_exact):7.1f}% {f(hit_sim):8.1f}% {f(hit_hom):10.1f}% {f(either):10.1f}%")

# ---------- (3) BigBind distant-homology per-target ----------
print("\n" + "=" * 96)
print("(3) BigBind test targets whose protein has a 30%-cluster homolog in train")
print("=" * 96)
cl30 = (edges.filter(pl.col("edge_type") == "protein_cluster_30")
        .select(pl.col("src").alias("prot"), pl.col("dst").alias("cl")).collect())
bb = eis.filter(pl.col("corpus") == "BigBind").join(ehp, on="ex")
train_cl = bb.filter(pl.col("part").is_in(["train", "val"])).join(cl30, on="prot")["cl"].unique().to_list()
test_t = bb.filter(pl.col("part") == "test").join(cl30, on="prot")
test_t = test_t.with_columns(pl.col("cl").is_in(train_cl).alias("leaks"))
per = (test_t.group_by("prot").agg(pl.col("leaks").first(), pl.len().alias("n_ex"))
       .sort("n_ex", descending=True))
nleak = per.filter(pl.col("leaks")).height
print(f"   {nleak}/{per.height} test proteins have a 30% homolog in train; top by #examples:")
print(per.head(12))
