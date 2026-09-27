"""Prove build_groups reproduces the audit_semantics block sizes, then exercise the
hub-capped regimes. Run: PYTHONNOUSERSITE=1 PYTHONPATH=src python tools/validate_leakage_groups.py
"""
from __future__ import annotations

import time

import polars as pl

from vsleakkg.split.leakage_groups import build_groups
from vsleakkg.split.regimes import REGIMES, Regime

NODES = "outputs/kg/canonical_nodes.parquet"
EDGES = "outputs/kg/canonical_edges.parquet"

nodes = pl.scan_parquet(NODES)
edges = pl.scan_parquet(EDGES)

print("=" * 90)
print("VALIDATION — cap=None regimes must match audit_semantics 'khối lớn nhất' exactly")
print("=" * 90)

# audit_semantics computes, at no cut:
#   ligand T>=0.80  |  scaffold cut=None  |  protein @30% (loosest, = my full protein regime)
checks = [
    Regime("ligand-clean", ("ligand",), degree_cap=None),
    Regime("scaffold-clean", ("scaffold",), degree_cap=None),
    Regime("protein-clean", ("protein",), degree_cap=None),
]
for reg in checks:
    t0 = time.time()
    res = build_groups(nodes, edges, reg)
    print(f"  {res.summary()}   [{time.time()-t0:.1f}s]")

print("\n" + "=" * 90)
print("HUB-CAPPED regimes (cap uniform over all content nodes; report, don't hide)")
print("=" * 90)
for name in ("scaffold-clean", "assay-clean", "dual-clean", "strict-clean"):
    reg = REGIMES[name]
    t0 = time.time()
    res = build_groups(nodes, edges, reg)
    flag = "  [label-confounded]" if reg.label_confounded else ""
    print(f"  {res.summary()}{flag}   [{time.time()-t0:.1f}s]")

# Sanity: every example assigned exactly once, ids contiguous.
reg = REGIMES["dual-clean"]
res = build_groups(nodes, edges, reg)
a = res.assignment
assert a["example_id"].n_unique() == a.height == res.n_examples, "duplicate/missing example"
assert a["group_id"].min() == 0 and a["group_id"].max() == res.n_groups - 1, "group ids not contiguous"
print(f"\nsanity: {res.n_examples:,} examples, each assigned once, group ids [0,{res.n_groups-1}] OK")
