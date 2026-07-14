"""Assemble node and edge parquet tables from a normalized examples frame.

The input frame must already contain at least:
  smiles_canonical (Utf8), scaffold_smiles (Utf8), source (Utf8), target (Utf8),
  label (Int8), label_type (Utf8), split (Utf8), example_id (Utf8).

`ligand_similar_to_ligand` edges are NOT built here — the audit module emits
them in a separate file because their size depends on the chosen Tanimoto
threshold and the chunking strategy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import polars as pl

from .graph_schema import EdgeType, NodeType


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def ligand_id(canonical_smiles: str) -> str:
    return f"lig:{_md5(canonical_smiles)}"


def scaffold_id(scaffold_smiles: str) -> str:
    # "" is a real value (acyclic / empty scaffold); we keep it as a distinct node.
    return f"sca:{_md5(scaffold_smiles or '')}"


def example_id(source: str, target: str, row_idx: int) -> str:
    return f"ex:{source}:{target}:{row_idx}"


def target_id(source: str, target: str) -> str:
    return f"tgt:{source}:{target}"


def source_id(source: str) -> str:
    return f"src:{source}"


def decoy_protocol_id(source: str) -> str:
    return f"prot:{source}"


def split_id(source: str, split_name: str) -> str:
    return f"split:{source}:{split_name}"


def label_type_id(label_type: str) -> str:
    return f"lt:{label_type}"


def build_examples_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add example_id, ligand_node_id, scaffold_node_id columns."""
    df = df.with_row_count("_row_idx").with_columns([
        pl.struct(["source", "target", "_row_idx"]).map_elements(
            lambda r: example_id(r["source"], r["target"], r["_row_idx"]),
            return_dtype=pl.Utf8, skip_nulls=False,
        ).alias("example_id"),
        pl.col("smiles_canonical").map_elements(
            lambda s: ligand_id(s) if s else None,
            return_dtype=pl.Utf8, skip_nulls=False,
        ).alias("ligand_node_id"),
        pl.col("scaffold_smiles").map_elements(
            lambda s: scaffold_id(s),
            return_dtype=pl.Utf8, skip_nulls=False,
        ).alias("scaffold_node_id"),
    ])
    return df


def make_nodes_edges(df: pl.DataFrame, *, include_decoy_protocol: bool,
                    include_protein_target: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the (nodes, edges) parquet-ready frames from an examples frame
    that already went through `build_examples_frame`."""
    nodes_rows = []
    edges_rows = []

    # Source / decoy-protocol / split / label-type / target nodes — small dim tables.
    for src in df["source"].unique().drop_nulls():
        nodes_rows.append((source_id(src), NodeType.DATASET_SOURCE.value, src, "{}"))
        if include_decoy_protocol:
            nodes_rows.append((decoy_protocol_id(src), NodeType.DECOY_PROTOCOL.value, src, "{}"))

    for split in df["split"].unique().drop_nulls():
        # source-scoped split nodes
        for src in df.filter(pl.col("split") == split)["source"].unique().drop_nulls():
            nodes_rows.append((split_id(src, split), NodeType.SPLIT.value, split, "{}"))

    for lt in df["label_type"].unique().drop_nulls():
        nodes_rows.append((label_type_id(lt), NodeType.LABEL_TYPE.value, lt, "{}"))

    if include_protein_target:
        for src, tgt in df.select(["source", "target"]).unique().drop_nulls().iter_rows():
            if tgt is None:
                continue
            nodes_rows.append((target_id(src, tgt), NodeType.PROTEIN_TARGET.value, tgt,
                               json.dumps({"source": src})))

    # Ligand nodes — one per unique canonical SMILES.
    lig_df = (df.filter(pl.col("smiles_canonical").is_not_null())
                .select(["ligand_node_id", "smiles_canonical", "scaffold_smiles", "scaffold_node_id"])
                .unique(subset=["ligand_node_id"]))
    for lid, csmi, scaf_smi, sid in lig_df.iter_rows():
        nodes_rows.append((lid, NodeType.LIGAND.value, csmi,
                           json.dumps({"scaffold_smiles": scaf_smi or ""})))
        edges_rows.append((lid, sid, EdgeType.LIGAND_HAS_SCAFFOLD.value, "{}"))

    # Scaffold nodes — one per unique scaffold (independent of ligand multiplicity).
    scaf_df = (df.select(["scaffold_node_id", "scaffold_smiles"])
                 .unique(subset=["scaffold_node_id"]))
    for sid, scaf_smi in scaf_df.iter_rows():
        nodes_rows.append((sid, NodeType.SCAFFOLD.value, scaf_smi or "",
                           json.dumps({"is_empty": scaf_smi in (None, "")})))

    # Example nodes + their outgoing edges.
    for row in df.iter_rows(named=True):
        eid = row["example_id"]
        nodes_rows.append((eid, NodeType.EXAMPLE.value, row.get("ext_id_1") or "",
                           json.dumps({
                               "label": int(row["label"]) if row["label"] is not None else None,
                               "target": row["target"],
                               "source": row["source"],
                           })))
        if row["ligand_node_id"]:
            edges_rows.append((eid, row["ligand_node_id"], EdgeType.EXAMPLE_HAS_LIGAND.value, "{}"))
        edges_rows.append((eid, source_id(row["source"]), EdgeType.EXAMPLE_FROM_SOURCE.value, "{}"))
        if row["split"] is not None:
            edges_rows.append((eid, split_id(row["source"], row["split"]),
                               EdgeType.EXAMPLE_IN_SPLIT.value, "{}"))
        if row["label_type"] is not None:
            edges_rows.append((eid, label_type_id(row["label_type"]),
                               EdgeType.EXAMPLE_HAS_LABEL_TYPE.value, "{}"))
        if include_protein_target and row["target"]:
            edges_rows.append((eid, target_id(row["source"], row["target"]),
                               EdgeType.EXAMPLE_TARGETS_PROTEIN.value, "{}"))
        if include_decoy_protocol:
            edges_rows.append((eid, decoy_protocol_id(row["source"]),
                               EdgeType.EXAMPLE_USES_DECOY_PROTOCOL.value, "{}"))

    nodes = pl.DataFrame(
        nodes_rows, schema=["node_id", "node_type", "label", "props"], orient="row"
    ).unique(subset=["node_id"])
    edges = pl.DataFrame(
        edges_rows, schema=["src", "dst", "edge_type", "props"], orient="row"
    ).unique()
    return nodes, edges
