"""Do the PUBLISHED splits leak? For each corpus that ships a real train/eval split
(LIT-PCBA=AVE-unbiased, BigBind, BayesBind), measure the fraction of eval examples
that sit in the same leakage group as a training example, per axis and for the
high-order union. A straddling group = leakage the split's own method did not catch.

The KG's whole thesis in one table: each prior method controls one modality and is
blind to the others + to multi-hop typed paths.

Run: PYTHONNOUSERSITE=1 POLARS_MAX_THREADS=1 PYTHONPATH=src python audit_published_splits.py
"""
from __future__ import annotations
import time
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

NODES = "outputs/kg/canonical_nodes.parquet"
EDGES = "outputs/kg/canonical_edges.parquet"
nodes = pl.scan_parquet(NODES)
edges = pl.scan_parquet(EDGES)

examples = nodes.filter(pl.col("node_type") == "Example").select(pl.col("node_id").alias("ex")).collect()
degree = nodes.select("node_id", "degree").collect()
DEG = dict(zip(degree["node_id"], degree["degree"]))

# example -> (corpus, partition)
eis = (
    edges.filter(pl.col("edge_type") == "example_in_split")
    .select(pl.col("src").alias("ex"), pl.col("dst").alias("split"))
    .collect()
    .with_columns(
        pl.col("split").str.split(":").list.get(1).alias("corpus"),
        pl.col("split").str.split(":").list.get(2).alias("part"),
    )
)

# corpus -> (eval_part, seen_parts)
SPLIT_ROLE = {
    "LIT-PCBA": ("validation", {"train"}),
    "BigBind": ("test", {"train", "val"}),
    "BayesBind": ("test", {"val"}),
}


def group_labels(edge_types: list[str], tmin: float | None, cap: int | None) -> pl.DataFrame:
    """example_id -> group_id over the subgraph kept by edge_types (whole KG)."""
    sub = edges.filter(pl.col("edge_type").is_in(edge_types))
    if tmin is not None:
        sub = sub.filter(
            (pl.col("edge_type") != "ligand_similar")
            | (pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64) >= tmin)
        )
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
        g = coo_matrix((np.ones(len(si), np.int8), (si, di)), shape=(nn, nn))
        _, lab = connected_components(g, directed=False)
    else:
        lab = np.arange(nn)
    ex_i = examples.join(idx, left_on="ex", right_on="id")["i"].to_numpy()
    return examples.with_columns(pl.Series("group_id", lab[ex_i]))


def leak_pct(labels: pl.DataFrame) -> dict[str, float]:
    """For each corpus: % of eval examples whose group holds a 'seen' example."""
    d = eis.join(labels, on="ex")
    out = {}
    for corpus, (evalp, seen) in SPLIT_ROLE.items():
        c = d.filter(pl.col("corpus") == corpus)
        # groups that contain a 'seen' (train/val) example of this corpus
        seen_groups = set(c.filter(pl.col("part").is_in(list(seen)))["group_id"].to_list())
        ev = c.filter(pl.col("part") == evalp)
        if not ev.height:
            out[corpus] = float("nan"); continue
        leaked = ev.filter(pl.col("group_id").is_in(seen_groups)).height
        out[corpus] = 100 * leaked / ev.height
    return out


# ---- regimes to audit (granular; cap chosen to avoid trivial-hub inflation) ----
AUDIT = [
    ("ligand-identity", ["example_has_ligand", "ligand_exact", "ligand_parent_exact",
                         "ligand_fingerprint_exact"], None, None),
    ("ligand+sim0.80", ["example_has_ligand", "ligand_exact", "ligand_parent_exact",
                        "ligand_fingerprint_exact", "ligand_similar"], 0.80, None),
    ("scaffold(cap1k)", ["example_has_ligand", "ligand_scaffold"], None, 1000),
    ("protein-identity", ["example_has_protein", "protein_exact"], None, None),
    ("protein-clust90", ["example_has_protein", "protein_exact", "protein_cluster_90"], None, None),
    ("protein-clust30", ["example_has_protein", "protein_exact", "protein_cluster_30"], None, None),
    ("assay(cap1k)", ["example_from_assay", "example_from_publication"], None, 1000),
    ("HIGH-ORDER all", ["example_has_ligand", "ligand_exact", "ligand_parent_exact",
                        "ligand_fingerprint_exact", "ligand_similar", "ligand_scaffold",
                        "example_has_protein", "protein_exact", "protein_cluster_90"], 0.80, 1000),
]

print("Leakage of PUBLISHED splits: % of EVAL examples sharing a leakage group with TRAIN")
print("(LIT-PCBA eval=validation | BigBind eval=test | BayesBind eval=test)")
print("=" * 92)
print(f"{'axis / regime':<20} {'LIT-PCBA':>12} {'BigBind':>12} {'BayesBind':>12}")
print("-" * 92)
for name, ets, tmin, cap in AUDIT:
    t0 = time.time()
    lab = group_labels(ets, tmin, cap)
    lp = leak_pct(lab)
    print(f"{name:<20} {lp['LIT-PCBA']:>11.1f}% {lp['BigBind']:>11.1f}% {lp['BayesBind']:>11.1f}%   [{time.time()-t0:.0f}s]")

# ---- cross-modal pretraining leakage: eval (ligand,protein) already measured in ChEMBL/BindingDB
print("-" * 92)
lmp = (edges.filter(pl.col("edge_type") == "ligand_measured_protein")
       .select(pl.col("src").alias("lig"), pl.col("dst").alias("prot")).collect().unique())
ehl = (edges.filter(pl.col("edge_type") == "example_has_ligand")
       .select(pl.col("src").alias("ex"), pl.col("dst").alias("lig")).collect())
ehp = (edges.filter(pl.col("edge_type") == "example_has_protein")
       .select(pl.col("src").alias("ex"), pl.col("dst").alias("prot")).collect())
expair = ehl.join(ehp, on="ex")
measured = expair.join(lmp, on=["lig", "prot"]).select("ex").unique().with_columns(pl.lit(1).alias("m"))
dd = eis.join(measured, on="ex", how="left")
print("cross-modal: % of EVAL (ligand,protein) already measured in ChEMBL/BindingDB (ligand_measured_protein)")
for corpus, (evalp, seen) in SPLIT_ROLE.items():
    ev = dd.filter((pl.col("corpus") == corpus) & (pl.col("part") == evalp))
    pct = 100 * ev.filter(pl.col("m") == 1).height / ev.height if ev.height else float("nan")
    print(f"   {corpus:<12} {pct:6.1f}%")
