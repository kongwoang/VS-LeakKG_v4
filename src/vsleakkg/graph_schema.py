"""Node and edge type definitions for the VS-LeakKG MVP graph.

Node tables share columns: node_id (str), node_type (str), label (str), props (Utf8 JSON).
Edge tables share columns: src (str), dst (str), edge_type (str), props (Utf8 JSON).

Node IDs are typed prefixes so the union of node tables remains a flat namespace:
  ex:<source>:<target>:<row_index>  Example  — FOUR fields, not three. The docstring
                                     said three; `build_graph.example_id()` has always
                                     written four, and `fixes.split_example_protein_
                                     relation` PARSES fields 1 and 2 to recover the
                                     target. Anyone who trusted this line and dropped
                                     the target field would silently break the protein
                                     axis for every corpus.
  lig:<canonical_smiles_md5>       Ligand
  sca:<scaffold_smiles_md5>        Scaffold
  tgt:<source>:<target_name>       ProteinTarget
  src:<source>                     DatasetSource
  prot:<source>                    DecoyProtocol
  split:<source>:<split_name>      Split
  lt:<active|decoy|inactive>       LabelType
  assay:<source>:<assay_id>        Assay (only if assay metadata available)
"""
from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    EXAMPLE = "Example"
    LIGAND = "Ligand"
    SCAFFOLD = "Scaffold"
    PROTEIN_TARGET = "ProteinTarget"
    DATASET_SOURCE = "DatasetSource"
    DECOY_PROTOCOL = "DecoyProtocol"
    SPLIT = "Split"
    LABEL_TYPE = "LabelType"
    ASSAY = "Assay"


class EdgeType(str, Enum):
    EXAMPLE_HAS_LIGAND = "example_has_ligand"
    LIGAND_HAS_SCAFFOLD = "ligand_has_scaffold"
    EXAMPLE_FROM_SOURCE = "example_from_source"
    EXAMPLE_IN_SPLIT = "example_in_split"
    EXAMPLE_TARGETS_PROTEIN = "example_targets_protein"
    EXAMPLE_USES_DECOY_PROTOCOL = "example_uses_decoy_protocol"
    EXAMPLE_HAS_LABEL_TYPE = "example_has_label_type"
    EXAMPLE_FROM_ASSAY = "example_from_assay"
    LIGAND_SIMILAR_TO_LIGAND = "ligand_similar_to_ligand"


NODE_SCHEMA = {"node_id": "Utf8", "node_type": "Utf8", "label": "Utf8", "props": "Utf8"}
EDGE_SCHEMA = {"src": "Utf8", "dst": "Utf8", "edge_type": "Utf8", "props": "Utf8"}
