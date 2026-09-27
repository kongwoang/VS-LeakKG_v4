"""Is the 'hard tier' (exact-identity only) free of a giant component? If yes, the
two-tier split is feasible: hard-merge identity, minimise everything else softly.
Run: PYTHONNOUSERSITE=1 POLARS_MAX_THREADS=1 PYTHONPATH=src python probe_hardtier.py
"""
from __future__ import annotations
import time
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

nodes = pl.scan_parquet("outputs/kg/canonical_nodes.parquet")
edges = pl.scan_parquet("outputs/kg/canonical_edges.parquet")
examples = nodes.filter(pl.col("node_type") == "Example").select(pl.col("node_id").alias("id")).collect()
N = examples.height


def largest(edge_types: list[str], name: str) -> None:
    t0 = time.time()
    pairs = edges.filter(pl.col("edge_type").is_in(edge_types)).select("src", "dst").collect()
    universe = pl.concat([examples["id"], pairs["src"], pairs["dst"]]).unique().sort()
    idx = pl.DataFrame({"id": universe}).with_row_index("i")
    e2 = (pairs.join(idx.rename({"id": "src", "i": "si"}), on="src")
                .join(idx.rename({"id": "dst", "i": "di"}), on="dst"))
    si, di = e2["si"].to_numpy(), e2["di"].to_numpy()
    n_nodes = idx.height
    g = coo_matrix((np.ones(len(si), np.int8), (si, di)), shape=(n_nodes, n_nodes))
    _, lab = connected_components(g, directed=False)
    ex_i = examples.join(idx, on="id")["i"].to_numpy()
    _, cnt = np.unique(lab[ex_i], return_counts=True)
    big = int(cnt.max())
    print(f"  {name:<38} {len(cnt):>9,} groups | largest {big:>9,} = {100*big/N:5.2f}% | edges {len(si):>10,} [{time.time()-t0:.0f}s]")


print(f"Example: {N:,}\n" + "=" * 90)
largest(["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact"],
        "ligand-identity (exact+parent+fp)")
largest(["example_has_protein", "protein_exact"], "protein-identity (exact)")
largest(["example_has_ligand", "ligand_exact", "ligand_parent_exact", "ligand_fingerprint_exact",
         "example_has_protein", "protein_exact"], "HARD TIER = ligand-id UNION protein-id")
