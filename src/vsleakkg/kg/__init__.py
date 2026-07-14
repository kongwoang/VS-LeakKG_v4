"""VS-LeakKG canonical knowledge-graph schema.

This package defines the canonical (axis-aligned) KG schema that downstream
audit consumes:

- `schema.py`     — NodeType / EdgeType enums, leakage-axis decomposition,
                    default edge weights, hub-mitigation defaults.
- `consolidate.py` — read the raw KG produced by `vsleakkg.build_kg`
                    (data/processed/kg_nodes.parquet + kg_edges.parquet),
                    apply schema mapping, hub flag, trivial scaffold drop,
                    protein-cluster enrichment, and write the canonical
                    parquet pair under outputs/kg/.
- `hydrate.py`, `build_side_table.py`, `trainset.py` — placeholders for
                    the model-adapter and Mode-B audit hooks (not yet
                    in active use in v3).

Pipeline:

  data/raw/...
      ↓ (loaders, build_kg)
  data/processed/kg_nodes.parquet, kg_edges.parquet         (raw KG)
      ↓ (consolidate)
  outputs/kg/canonical_nodes.parquet, canonical_edges.parquet
"""
from __future__ import annotations

__version__ = "3.0.0-dev"
