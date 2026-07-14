"""Canonical KG schema — node / edge types and default leakage weights.

The weights match proposal.tex Table 2 (the "Default edge types and leakage
weights" table). Edit DEFAULT_WEIGHTS to change a release-wide default and bump
the version string in __init__.py. For run-time overrides, pass a dict to
score_axis() / score_overall() in scoring.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NodeType(str, Enum):
    EXAMPLE = "Example"
    PROTEIN = "Protein"
    PROTEIN_CLUSTER = "ProteinCluster"  # per-resolution; props["resolution"] = "30" | "40" | "90"
    LIGAND = "Ligand"
    SCAFFOLD = "Scaffold"
    ASSAY = "Assay"
    PUBLICATION = "Publication"
    DATASET_SOURCE = "DatasetSource"
    DECOY_PROTOCOL = "DecoyProtocol"
    # The *method* a decoy protocol implements. Two tiers, because "DUD-E and
    # DEKOIS both use property-matched decoys" is an INTERPRETATION, not a fact the
    # data states. The graph records both facts separately — the corpus's own
    # protocol, and which method class it belongs to — and leaves it to the
    # downstream step to decide whether to traverse the class tier.
    DECOY_PROTOCOL_CLASS = "DecoyProtocolClass"
    TIMEBIN = "TimeBin"
    TRAINSET = "TrainSet"  # Mode B: model-specific; props["model"] = "<model_id>"


class EdgeType(str, Enum):
    # binding edges connecting an Example to its constituent entities
    EXAMPLE_HAS_LIGAND = "example_has_ligand"
    EXAMPLE_HAS_PROTEIN = "example_has_protein"
    EXAMPLE_FROM_ASSAY = "example_from_assay"
    EXAMPLE_FROM_PUBLICATION = "example_from_publication"
    EXAMPLE_FROM_SOURCE = "example_from_source"
    EXAMPLE_HAS_TIMEBIN = "example_has_timebin"
    EXAMPLE_IN_TRAINSET = "example_in_trainset"  # Mode B

    # identity / similarity edges between content nodes
    LIGAND_EXACT = "ligand_exact"               # same full InChIKey, different SMILES
    # Same PARENT InChIKey — salt-stripped, charge-neutralised, stereo-free — and a
    # DIFFERENT full InChIKey. Both halves of that sentence were false before:
    # `chem.parent_inchikey` only stripped salts (so a protonated amine never matched
    # its neutral twin), and `build_kg` emitted the edge for every parent group
    # including those already sharing a full key. Result: the relation was a 100 %
    # duplicate of LIGAND_EXACT (6,939 == 6,939 identical pairs) and caught zero of
    # the variants it exists for, while 41,238 pairs of Ligand nodes that are the same
    # compound sat in the graph with no edge between them.
    LIGAND_PARENT_EXACT = "ligand_parent_exact"
    LIGAND_FINGERPRINT_EXACT = "ligand_fingerprint_exact"   # ECFP4 Tanimoto = 1.0, different SMILES (typically stereo)
    LIGAND_SCAFFOLD = "ligand_scaffold"
    # ECFP4 Tanimoto in [0.80, 0.9995); >= 0.9995 is LIGAND_FINGERPRINT_EXACT.
    # The 0.80 floor is now real. It used to be a claim: the global pass had been run
    # at 0.85, and the band [0.80, 0.85) held 427 edges — all of them LIT-PCBA legacy
    # rows — where the shape of the distribution says it should hold hundreds of
    # thousands. Changing this floor means changing `ligand_similarity.py` and the
    # docs in the same commit, and rebuilding.
    LIGAND_SIMILAR = "ligand_similar"
    PROTEIN_EXACT = "protein_exact"
    # Ligand -> Protein, from BindingDB/ChEMBL: "this ligand has a measured activity
    # against this protein". Deliberately NOT in AXIS_EDGE_TYPES. It used to be
    # emitted as example_has_protein, which conflated "the example's TARGET is P"
    # with "the example's LIGAND was once measured against P" — that turned a
    # ligand-mediated path into a protein-axis path and collapsed the protein axis
    # into a single component covering 100 % of examples. Kept as its own relation
    # because it is genuine evidence of pretraining contamination (a ChEMBL/BindingDB
    # -trained model has seen the (ligand, protein) pair), just not benchmark-split
    # leakage.
    LIGAND_MEASURED_PROTEIN = "ligand_measured_protein"
    # One edge type per MMseqs2 resolution: the leakage weight of "same cluster"
    # depends entirely on how tight the cluster is, and a single edge type forced
    # every scorer to reach into props to find out — which is how the axis ended
    # up with no weight at all (see docs/kg_audit_2026-07-14.md).
    PROTEIN_CLUSTER_30 = "protein_cluster_30"   # 30 % sequence identity — distant family
    PROTEIN_CLUSTER_50 = "protein_cluster_50"   # 50 % — clear homolog
    PROTEIN_CLUSTER_90 = "protein_cluster_90"   # 90 % — effectively the same protein
    SOURCE_DECOY_PROTOCOL = "source_decoy_protocol"
    DECOY_PROTOCOL_IN_CLASS = "decoy_protocol_in_class"   # DecoyProtocol -> DecoyProtocolClass
    TIME_OVERLAP = "time_overlap"


# Default edge weights mirror proposal.tex Table 2.
# Weights are in (0, 1]; 1.0 means "exact identity / strongest possible leak."
DEFAULT_WEIGHTS: dict[str, float] = {
    EdgeType.EXAMPLE_HAS_LIGAND.value: 1.00,
    EdgeType.LIGAND_EXACT.value: 1.00,
    EdgeType.LIGAND_PARENT_EXACT.value: 0.95,         # salt/protonation variant — near-identical leak
    EdgeType.LIGAND_FINGERPRINT_EXACT.value: 0.95,    # ECFP4 = same: stereo / tautomer the fp can't see
    EdgeType.LIGAND_SCAFFOLD.value: 0.70,
    EdgeType.LIGAND_SIMILAR.value: 0.65,
    EdgeType.EXAMPLE_HAS_PROTEIN.value: 1.00,
    EdgeType.PROTEIN_EXACT.value: 1.00,
    EdgeType.LIGAND_MEASURED_PROTEIN.value: 0.90,   # not in any axis; see EdgeType
    EdgeType.PROTEIN_CLUSTER_90.value: 0.85,
    EdgeType.PROTEIN_CLUSTER_50.value: 0.65,
    EdgeType.PROTEIN_CLUSTER_30.value: 0.45,
    EdgeType.EXAMPLE_FROM_ASSAY.value: 0.75,
    EdgeType.EXAMPLE_FROM_PUBLICATION.value: 0.55,
    EdgeType.EXAMPLE_FROM_SOURCE.value: 0.35,
    EdgeType.SOURCE_DECOY_PROTOCOL.value: 0.50,
    EdgeType.DECOY_PROTOCOL_IN_CLASS.value: 0.90,
    EdgeType.TIME_OVERLAP.value: 0.40,
    EdgeType.EXAMPLE_IN_TRAINSET.value: 1.00,
    EdgeType.EXAMPLE_HAS_TIMEBIN.value: 1.00,
}


# Axis subgraphs: each axis is computed on its own subgraph (proposal section 5.5).
# An axis-specific subgraph uses example_has_* binding edges PLUS the relational
# edges listed for that axis. This ensures per-axis decomposition is well-defined
# even when paths could mix multiple edge types.
AXIS_EDGE_TYPES: dict[str, list[str]] = {
    "ligand": [
        EdgeType.EXAMPLE_HAS_LIGAND.value,
        EdgeType.LIGAND_EXACT.value,
        EdgeType.LIGAND_PARENT_EXACT.value,
        EdgeType.LIGAND_FINGERPRINT_EXACT.value,
        EdgeType.LIGAND_SIMILAR.value,
    ],
    "scaffold": [
        EdgeType.EXAMPLE_HAS_LIGAND.value,
        EdgeType.LIGAND_SCAFFOLD.value,
    ],
    "protein": [
        EdgeType.EXAMPLE_HAS_PROTEIN.value,
        EdgeType.PROTEIN_EXACT.value,
        EdgeType.PROTEIN_CLUSTER_30.value,
        EdgeType.PROTEIN_CLUSTER_50.value,
        EdgeType.PROTEIN_CLUSTER_90.value,
    ],
    "assay": [
        EdgeType.EXAMPLE_FROM_ASSAY.value,
        EdgeType.EXAMPLE_FROM_PUBLICATION.value,
    ],
    "source": [
        EdgeType.EXAMPLE_FROM_SOURCE.value,
        EdgeType.SOURCE_DECOY_PROTOCOL.value,
        EdgeType.DECOY_PROTOCOL_IN_CLASS.value,
    ],
    "time": [
        EdgeType.EXAMPLE_HAS_TIMEBIN.value,
        EdgeType.TIME_OVERLAP.value,
    ],
}

AXES: tuple[str, ...] = tuple(AXIS_EDGE_TYPES.keys())


@dataclass(frozen=True)
class HubMitigationConfig:
    """Hub-pollution mitigation parameters (proposal section 5.3).

    - trivial_scaffold_max_atoms: scaffolds with <= this many heavy atoms (and
      no substituents) are dropped from the scaffold axis. Default 6 = single
      ring like benzene with no chains.
    - degree_cap: nodes with degree > cap are split into per-source shards.
    - idf_floor: minimum weight after IDF downweighting (relative to nominal).
    """
    trivial_scaffold_max_atoms: int = 6
    degree_cap: int = 1000
    idf_floor: float = 0.10


@dataclass(frozen=True)
class GiantComponentConfig:
    """Giant-component fallback thresholds (proposal section 5.9)."""
    rho_max_ok: float = 0.30
    rho_max_prune: float = 0.60
    # Above rho_max_prune we fall back to Louvain community detection.


@dataclass(frozen=True)
class SplitConstraints:
    """Default group-assignment constraints (proposal section 5.10)."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    label_balance_tol: float = 0.05  # |D_k+|/|D_k| - |D+|/|D|
    min_targets_per_partition: int = 5
    min_actives_per_partition: int = 20
    lambda_size: float = 1.0
    lambda_label: float = 1.0
    lambda_cover: float = 0.5
    lambda_resid: float = 1.0
