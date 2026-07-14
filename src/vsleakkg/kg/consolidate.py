"""Consolidate the raw KG into the canonical audit-ready schema.

Reads the raw KG produced by `vsleakkg.build_kg`:

  data/processed/kg_nodes.parquet
  data/processed/kg_edges.parquet
  data/processed/protein_clusters_{30,50,90}.parquet  (sequence-axis anchor,
                                                       optional)

Applies:
  1. Schema mapping     — collapse corpus-level type names (e.g. ChEMBLAssay
                          → Assay, ProteinTarget → Protein) onto the canonical
                          set defined in `vsleakkg.kg.schema`.
  2. Lossy node drop    — discard pure scaffolding (ChEMBLActivity, Split,
                          LabelType, DatabaseRelease) that the audit doesn't
                          need.
  3. Protein clustering — emit ProteinCluster nodes + protein_in_cluster edges
                          from the optional sequence-clustered parquets.
  4. Hub mitigation     — flag any node with degree > HubMitigationConfig.
                          degree_cap (default 1000) as `is_hub`.
  5. Trivial scaffold   — drop scaffolds with ≤ trivial_scaffold_max_atoms
                          (default 6) heavy atoms.

Outputs:

  outputs/kg/canonical_nodes.parquet
  outputs/kg/canonical_edges.parquet
  outputs/kg/stats.csv

CLI:

    python -m vsleakkg.kg.consolidate \\
        --output-dir outputs/kg \\
        [--corpus litpcba_ave|dude|dekois|bigbind|bayesbind|all]   # default all
        [--limit 100000]    # for smoke-testing

The script is idempotent: it will overwrite the output parquets.
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from .datapaths import processed_dir, require_data_root
from .schema import (
    AXIS_EDGE_TYPES,
    DEFAULT_WEIGHTS,
    EdgeType,
    HubMitigationConfig,
    NodeType,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus-level → canonical schema mapping tables
# ---------------------------------------------------------------------------
# `build_kg` emits the raw KG using the corpus-level node/edge type names
# inherited from the per-corpus loaders (`Ligand`, `ProteinTarget`,
# `ChEMBLAssay`, ...). The canonical schema (`vsleakkg.kg.schema`) collapses
# semantically-equivalent types onto a smaller set. These maps drive the
# rewrite, with the side effect that certain pure-scaffolding node/edge types
# are dropped entirely.

CORPUS_TO_CANONICAL_NODE_TYPE: dict[str, str] = {
    "Example": NodeType.EXAMPLE.value,
    "Ligand": NodeType.LIGAND.value,
    "Scaffold": NodeType.SCAFFOLD.value,
    "Protein": NodeType.PROTEIN.value,
    "ProteinTarget": NodeType.PROTEIN.value,
    "ChEMBLAssay": NodeType.ASSAY.value,
    "Assay": NodeType.ASSAY.value,                # identity: BindingDB-emitted Assay-like records
    "ChEMBLDocument": NodeType.PUBLICATION.value,
    "Publication": NodeType.PUBLICATION.value,    # identity: BindingDB-emitted PMID/DOI publications
    "DatasetSource": NodeType.DATASET_SOURCE.value,
    "DecoyProtocol": NodeType.DECOY_PROTOCOL.value,
    "Split": NodeType.SPLIT.value,            # the corpus's own published split
    "LabelType": NodeType.LABEL_TYPE.value,   # active / decoy / inactive / random
}

# Corpus-level node types that are absorbed elsewhere or are pure scaffolding.
#
# `LabelType` and `Split` used to be here. Both justifications were false, and both
# were false in the same way: they named a mechanism that does not exist.
#
#   "Split — canonical schema emits partition assignments separately"
#       It does not. There is no partition-assignment code, no Split/Partition node
#       type, no scoring.py. 5,025,497 example_in_split edges — one per Example,
#       100 % coverage — were deleted, taking LIT-PCBA's AVE_unbiased train/
#       validation split and BayesBind's val/test split with them. AVE debiasing is
#       the strongest published defence against benchmark bias; "contamination
#       survives AVE" is the single most valuable experiment this KG can support,
#       and the canonical graph had thrown away the ability to state it.
#
#   "LabelType — static lookup table"
#       A lookup table is a mapping you can rebuild from nothing. This one carries
#       the corpus's own assertion about each example, and `label` (0/1) does not
#       encode it: a DUD-E property-matched decoy, a LIT-PCBA measured inactive,
#       and a BayesBind random molecule are all `label = 0` and were, in the
#       canonical graph, indistinguishable. That is the exact distinction this
#       project exists to interrogate.
#
# Both are now kept as facts and placed in NO axis (schema.NON_AXIS_EDGE_TYPES).
DROPPED_NODES: frozenset[str] = frozenset({
    "ChEMBLActivity",        # absorbed into Example via label + label_type props
    "AffinityType",          # static lookup table
    "DatabaseRelease",       # version metadata
})

CORPUS_TO_CANONICAL_EDGE_TYPE: dict[str, str] = {
    "example_has_ligand": EdgeType.EXAMPLE_HAS_LIGAND.value,
    "example_targets_protein": EdgeType.EXAMPLE_HAS_PROTEIN.value,
    "example_from_source": EdgeType.EXAMPLE_FROM_SOURCE.value,
    "ligand_has_scaffold": EdgeType.LIGAND_SCAFFOLD.value,
    "ligand_similar_to_ligand": EdgeType.LIGAND_SIMILAR.value,
    "ligand_similar": EdgeType.LIGAND_SIMILAR.value,    # ligand_similarity.py emits this name directly
    "ligand_fingerprint_exact": EdgeType.LIGAND_FINGERPRINT_EXACT.value,
    "same_inchikey_as": EdgeType.LIGAND_EXACT.value,
    "same_parent_inchikey_as": EdgeType.LIGAND_PARENT_EXACT.value,
    "example_uses_decoy_protocol": EdgeType.SOURCE_DECOY_PROTOCOL.value,
    # Facts, in no axis. See DROPPED_NODES above and schema.NON_AXIS_EDGE_TYPES.
    "example_in_split": EdgeType.EXAMPLE_IN_SPLIT.value,
    "example_has_label_type": EdgeType.EXAMPLE_HAS_LABEL_TYPE.value,
}
# BindingDB enrichment edges (bdb_lig -> Publication / Protein / Assay /
# bdb_rec -> Ligand / Protein) are NOT mapped here: they form 2-hop paths
# through `bdb_lig` from benchmark Example -> Ligand -> bdb_lig, which the
# canonical single-edge axis schema can't represent. They remain in the raw
# kg_edges parquet for downstream graph-traversal queries that want the
# full BindingDB provenance.

# Nothing is dropped any more. Kept as an explicit empty set, with the history, so
# that adding a name back here is a deliberate act and not a passing convenience.
DROPPED_EDGES: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    n_nodes_in: int = 0
    n_edges_in: int = 0
    n_nodes_dropped: int = 0
    n_edges_dropped: int = 0
    n_nodes_out: int = 0
    n_edges_out: int = 0
    nodes_by_type: dict[str, int] = None     # type: ignore[assignment]
    edges_by_type: dict[str, int] = None     # type: ignore[assignment]
    n_protein_cluster_edges: dict[str, int] = None  # type: ignore[assignment]
    n_trivial_scaffolds_dropped: int = 0
    n_hub_nodes_sharded: int = 0
    deferred: list[str] = None    # type: ignore[assignment]

    def to_csv_rows(self) -> list[dict]:
        rows: list[dict] = [
            {"key": "n_nodes_in", "value": self.n_nodes_in},
            {"key": "n_edges_in", "value": self.n_edges_in},
            {"key": "n_nodes_dropped", "value": self.n_nodes_dropped},
            {"key": "n_edges_dropped", "value": self.n_edges_dropped},
            {"key": "n_nodes_out", "value": self.n_nodes_out},
            {"key": "n_edges_out", "value": self.n_edges_out},
            {"key": "n_trivial_scaffolds_dropped",
             "value": self.n_trivial_scaffolds_dropped},
            {"key": "n_hub_nodes_sharded", "value": self.n_hub_nodes_sharded},
        ]
        for t, n in sorted((self.nodes_by_type or {}).items()):
            rows.append({"key": f"nodes_by_type::{t}", "value": int(n)})
        for t, n in sorted((self.edges_by_type or {}).items()):
            rows.append({"key": f"edges_by_type::{t}", "value": int(n)})
        for res, n in sorted((self.n_protein_cluster_edges or {}).items()):
            rows.append({"key": f"protein_cluster_edges::{res}", "value": int(n)})
        for d in self.deferred or []:
            rows.append({"key": f"deferred::{d}", "value": 0})
        return rows


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------


from vsleakkg.kg import fixes as _fixes


def _map_nodes(nodes: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Map corpus node_type to canonical node_type. Drop scaffolding-only types."""
    keep = list(CORPUS_TO_CANONICAL_NODE_TYPE.keys())
    n_in = nodes.height
    kept = nodes.filter(pl.col("node_type").is_in(keep)).with_columns(
        pl.col("node_type").replace(CORPUS_TO_CANONICAL_NODE_TYPE).alias("node_type")
    )
    return kept, n_in - kept.height


def _enrich_example_props(nodes: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """Fold `label_type` and `split` into every Example's props.

    Both are already in the graph as edges (Example -> lt:*, Example -> split:*),
    so this is redundancy, not new information — deliberately. An Example's props
    already duplicate `label`, `target` and `source` even though example_from_source
    exists; the one field that mattered most was the one left out. Reading a label
    type should not require a join, because a step that requires a join is a step
    that gets skipped.

    Totality is asserted, not hoped for: every Example must come out with both. If a
    corpus ever ships an Example without a label type, this raises here rather than
    letting a null propagate into a contamination score.
    """
    ex = nodes.filter(pl.col("node_type") == NodeType.EXAMPLE.value)
    if ex.is_empty():
        return nodes

    def _attr(edge_type: str, prefix: str, name: str) -> pl.DataFrame:
        return (edges.filter(pl.col("edge_type") == edge_type)
                     .select(pl.col("src").alias("node_id"),
                             pl.col("dst").str.strip_prefix(prefix).alias(name))
                     .unique(subset=["node_id"]))

    lt = _attr(EdgeType.EXAMPLE_HAS_LABEL_TYPE.value, "lt:", "label_type")
    sp = _attr(EdgeType.EXAMPLE_IN_SPLIT.value, "split:", "split")

    merged = (ex.select("node_id", "props")
                .join(lt, on="node_id", how="left")
                .join(sp, on="node_id", how="left"))

    missing_lt = int(merged["label_type"].null_count())
    missing_sp = int(merged["split"].null_count())
    if missing_lt or missing_sp:
        raise ValueError(
            f"Example props enrichment incomplete: {missing_lt} without a label_type, "
            f"{missing_sp} without a split. Every Example must have both — a corpus "
            f"that ships neither is a corpus whose negatives cannot be interpreted."
        )

    # The props JSON is extended textually — cheap at 5 M rows, but it assumes props
    # is a non-empty object. `{}` would splice into `{, "label_type": ...}`: invalid
    # JSON, and silently so, since nothing downstream re-parses it at write time.
    n_empty = int(merged.filter(pl.col("props").str.len_chars() < 3).height)
    if n_empty:
        raise ValueError(
            f"{n_empty} Examples have empty props — textual JSON splice would corrupt "
            f"them. Build the object properly if this ever becomes reachable."
        )

    # `split:LIT-PCBA:train` -> strip the prefix leaves `LIT-PCBA:train`; keep the
    # corpus qualifier, because "train" alone is ambiguous across five corpora.
    enriched = merged.with_columns(
        pl.concat_str([
            pl.col("props").str.strip_suffix("}"),
            pl.lit(', "label_type": "'), pl.col("label_type"),
            pl.lit('", "split": "'), pl.col("split"), pl.lit('"}'),
        ]).alias("props")
    ).select("node_id", "props")

    out = (nodes.join(enriched, on="node_id", how="left", suffix="_new")
                .with_columns(pl.coalesce(["props_new", "props"]).alias("props"))
                .drop("props_new"))
    log.info("Example props enriched with label_type + split (%d Examples)", ex.height)
    return out


def _map_edges(edges: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Map corpus edge_type to canonical edge_type. Drop scaffolding-only edges."""
    keep = list(CORPUS_TO_CANONICAL_EDGE_TYPE.keys())
    n_in = edges.height
    kept = edges.filter(pl.col("edge_type").is_in(keep)).with_columns(
        pl.col("edge_type").replace(CORPUS_TO_CANONICAL_EDGE_TYPE).alias("edge_type")
    )
    return kept, n_in - kept.height


# UniProt accession formats (https://www.uniprot.org/help/accession_numbers):
#   6-char: [OPQ][0-9][A-Z0-9]{3}[0-9]  OR  [A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]
#   10-char (extended, since 2014): [A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z][A-Z0-9]{2}[0-9]
_UNIPROT_RE = (
    r"^([OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]"
    r"|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9][A-Z][A-Z0-9]{2}[0-9])$"
)


def _normalize_protein_ids(
    nodes: pl.DataFrame, edges: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Collapse `tgt:Corpus:UniProtID` Protein nodes onto canonical
    `protein:UniProtID`.

    The per-corpus loaders prefix Protein node_ids with `tgt:<Corpus>:` even
    when the corpus already provides a clean UniProt accession (e.g. BigBind,
    BayesBind). BindingDB enrichment + cluster edges use the canonical
    `protein:<UniProt>` form. Without this normalisation the same UniProt
    becomes two distinct Protein nodes — ~1k overlap on BigBind alone —
    which splits the protein axis and silently weakens the audit.

    Strategy: detect node_ids matching `^tgt:[^:]+:<UniProtRegex>$`, rewrite
    them to `protein:<UniProtRegex>`, dedup Protein nodes by id (keep
    first row's props), then rewrite edge src/dst with the same mapping.
    """
    mapping = (
        nodes.filter(pl.col("node_type") == NodeType.PROTEIN.value)
        .with_columns(
            pl.col("node_id")
            .str.extract(r"^tgt:[^:]+:(.+)$", 1)
            .alias("_suffix")
        )
        .filter(
            pl.col("_suffix").is_not_null()
            & pl.col("_suffix").str.contains(_UNIPROT_RE)
        )
        .select(
            pl.col("node_id").alias("old_id"),
            (pl.lit("protein:") + pl.col("_suffix")).alias("new_id"),
        )
    )
    if not mapping.height:
        return nodes, edges, 0
    n_remapped = mapping.height
    # Apply to nodes: rewrite id, then drop duplicate ids (keep first).
    nodes_out = (
        nodes.join(mapping, left_on="node_id", right_on="old_id", how="left")
        .with_columns(
            pl.when(pl.col("new_id").is_not_null())
            .then(pl.col("new_id"))
            .otherwise(pl.col("node_id"))
            .alias("node_id")
        )
        .drop("new_id")
        .pipe(_fixes.stable_unique, ["node_id"])
    )
    # Apply to edges: rewrite src and dst the same way.
    edges_out = (
        edges.join(mapping, left_on="src", right_on="old_id", how="left")
        .with_columns(
            pl.when(pl.col("new_id").is_not_null())
            .then(pl.col("new_id"))
            .otherwise(pl.col("src"))
            .alias("src")
        )
        .drop("new_id")
        .join(mapping, left_on="dst", right_on="old_id", how="left")
        .with_columns(
            pl.when(pl.col("new_id").is_not_null())
            .then(pl.col("new_id"))
            .otherwise(pl.col("dst"))
            .alias("dst")
        )
        .drop("new_id")
    )
    return nodes_out, edges_out, n_remapped


def _drop_trivial_scaffolds(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    cfg: HubMitigationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Record each Scaffold's size; drop nothing.

    This used to delete scaffolds with <= 6 heavy atoms. But benzene IS a scaffold —
    calling it weak evidence of leakage is an interpretation, not a fact, and the KG
    has no business making it. The size is written to `props.n_heavy_atoms` so the
    downstream step can filter on whatever threshold it wants.

    The count itself used to be

        label.str.replace_all(r"[^A-Za-z]", "").str.len_chars()

    which counts LETTERS IN THE SMILES STRING, not atoms. `Cl` and `Br` counted 2;
    the H in `[nH]` and `[NH2+]` counted as an atom although hydrogen is not a heavy
    atom at all. Measured against RDKit on a 30,000-scaffold sample: 41.3 % of the
    values were wrong. This is the fact the removed trivial-scaffold filter was
    replaced BY — a downstream step filtering on `n_heavy_atoms <= 6` was filtering
    on a number that does not mean what it says. RDKit is already imported here.
    """
    import json as _j
    from rdkit import Chem as _Chem

    sc = nodes.filter(pl.col("node_type") == NodeType.SCAFFOLD.value)
    if not sc.height:
        return nodes, edges, 0

    def _n_heavy(smi: str) -> int:
        if not smi:
            return 0
        m = _Chem.MolFromSmiles(smi)
        if m is None:
            # A handful of RDKit Murcko artefacts (aromatic carbanions like [c-])
            # do not round-trip. Record -1 rather than a plausible-looking lie:
            # a downstream filter can see "unknown" but cannot see a wrong number.
            return -1
        return m.GetNumHeavyAtoms()

    def _with_size(row: dict) -> str:
        try:
            d = _j.loads(row["props"]) if row["props"] else {}
        except Exception:
            d = {}
        d["n_heavy_atoms"] = row["_n"]
        return _j.dumps(d, sort_keys=True)

    ann = (sc.with_columns(
               pl.col("label").map_elements(_n_heavy, return_dtype=pl.Int64).alias("_n"))
             .with_columns(
                 pl.struct(["props", "_n"])
                   .map_elements(_with_size, return_dtype=pl.Utf8).alias("_props"))
             .select("node_id", "_props"))
    nodes_out = (nodes.join(ann, on="node_id", how="left")
                      .with_columns(pl.coalesce([pl.col("_props"), pl.col("props")]).alias("props"))
                      .drop("_props"))
    return nodes_out, edges, 0


def _shard_hub_nodes(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    cfg: HubMitigationConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """Record each node's degree; make no judgement about it.

    This used to set `is_hub = degree > 1000`. That threshold is a mitigation
    POLICY, and 1000 is arbitrary — it belongs to whatever step decides how to score
    or forbid paths, not to the graph. "How informative is sharing this node" is a
    continuous quantity (an assay with 4 compounds is strong evidence; an HTS screen
    with 330,122 is none), and freezing it into a boolean throws that away. The KG
    now records the fact — the degree — and lets the downstream step pick its own
    threshold, or discount weights continuously instead of excluding at all.
    """
    deg_src = edges.group_by("src").agg(pl.len().alias("deg")).rename({"src": "node_id"})
    deg_dst = edges.group_by("dst").agg(pl.len().alias("deg")).rename({"dst": "node_id"})
    deg = (
        pl.concat([deg_src, deg_dst])
        .group_by("node_id")
        .agg(pl.col("deg").sum())
    )
    nodes_out = (nodes.join(deg, on="node_id", how="left")
                      .with_columns(pl.col("deg").fill_null(0).cast(pl.Int64).alias("degree"))
                      .drop("deg"))
    return nodes_out, edges, 0


def _add_protein_cluster_edges(
    edges: pl.DataFrame,
    nodes: pl.DataFrame,
    processed: Path,
    stats: BuildStats,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add protein_in_cluster edges from sequence-clustered parquets.

    Reads `protein_clusters_{30,50,90}.parquet` with columns (protein_id,
    cluster_id). Each cluster_id becomes a ProteinCluster node tagged with
    resolution = "30" | "50" | "90".

    The canonical schema weights are tuned for 30 / 50 / 90% sequence identity.
    Missing parquets are recorded in `stats.deferred` for the audit report.
    """
    new_node_dfs: list[pl.DataFrame] = []
    new_edge_dfs: list[pl.DataFrame] = []
    counts: dict[str, int] = {}
    for res in ("30", "50", "90"):
        f = processed / f"protein_clusters_{res}.parquet"
        if not f.exists():
            counts[res] = 0
            stats.deferred = (stats.deferred or []) + [f"protein_clusters_{res}_missing"]
            continue
        df = pl.read_parquet(f)
        # try to find protein-id and cluster-id columns
        col_protein = next(
            (c for c in df.columns
             if c.lower() in ("accession", "protein_id", "pdb_id", "member",
                              "sequence_id", "seq_id", "uniprot")),
            None,
        )
        col_cluster = next(
            (c for c in df.columns
             if c.lower() in ("cluster_id", "cluster", "representative",
                              "rep_seq", "rep_seq_id")),
            None,
        )
        if not col_protein or not col_cluster:
            counts[res] = 0
            stats.deferred = (stats.deferred or []) + [
                f"protein_clusters_{res}_unknown_schema_cols={df.columns}"
            ]
            continue
        df2 = df.select(
            # Prefix accession with `protein:` so the edge src matches the
            # Protein node id format used by the BindingDB enrichment step
            # in task_build_kg (`protein:<UniProt>`).
            (pl.lit("protein:") + pl.col(col_protein).cast(pl.Utf8)).alias("member_id"),
            pl.col(col_cluster).cast(pl.Utf8).alias("cluster_id"),
        )
        # Synthesise the ProteinCluster nodes (one per cluster_id).
        unique_clusters = df2["cluster_id"].unique().to_list()
        cluster_nodes = pl.DataFrame({
            "node_id":   [f"ProteinCluster::{res}::{c}" for c in unique_clusters],
            "node_type": [NodeType.PROTEIN_CLUSTER.value] * len(unique_clusters),
            "label":     [f"ProteinCluster::{res}::{c}" for c in unique_clusters],
            "props":     [f'{{"resolution":"{res}"}}'] * len(unique_clusters),
        })
        new_node_dfs.append(cluster_nodes)
        # Construct the protein_in_cluster edges (Protein -> ProteinCluster_<res>).
        edges_df = df2.with_columns(
            (pl.lit(f"ProteinCluster::{res}::") + pl.col("cluster_id").cast(pl.Utf8))
            .alias("dst"),
        ).select(
            pl.col("member_id").alias("src"),
            pl.col("dst"),
            pl.lit(getattr(EdgeType, f"PROTEIN_CLUSTER_{res}").value).alias("edge_type"),
            pl.lit(f'{{"resolution":"{res}"}}').alias("props"),
        )
        new_edge_dfs.append(edges_df)
        counts[res] = edges_df.height
    stats.n_protein_cluster_edges = counts
    if new_node_dfs:
        nodes = pl.concat([nodes] + new_node_dfs, how="vertical_relaxed")
    if new_edge_dfs:
        edges = pl.concat([edges] + new_edge_dfs, how="vertical_relaxed")
    return nodes, edges


def _add_protein_exact_edges(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    processed: Path,
    stats: BuildStats,
) -> pl.DataFrame:
    """Protein <-> Protein: same protein by sequence.

    `protein_exact` is declared in the schema, carries weight 1.00, and is listed in
    AXIS_EDGE_TYPES["protein"] — and the graph shipped with ZERO edges of it. So two
    Protein nodes holding the same protein under different accessions were never
    joined, and the strongest cross-corpus protein leak there is scored 0.85 x 0.85
    = 0.72 (the detour through the 90 % cluster) instead of 1.00. It also means the
    headline "142 proteins shared by >= 2 corpora" is an undercount: DUD-E and DEKOIS
    overlap on FGFR1, FKBP1A and JNK3, and the graph could not say so.

    The rule is structural, and deliberately says nothing about species: >= 98 %
    identity over >= 90 % of BOTH sequences (see tools/build_protein_axis.py). The
    both-sides coverage requirement is what preserves the HIV domain split — the
    99-aa protease is 100 % identical to a SLICE of the 1,447-aa polyprotein, but
    covers only 7 % of it.

    `props.pident` carries the identity, so a downstream step can raise the floor.
    The KG records the fact; the threshold is policy.
    """
    import json as _json

    f = processed / "protein_exact.parquet"
    if not f.exists():
        log.warning("protein_exact.parquet missing — the protein axis has no identity "
                    "relation; run tools/build_protein_axis.py")
        stats.deferred = (stats.deferred or []) + ["protein_exact_missing"]
        return edges
    pe = pl.read_parquet(f)
    if not pe.height:
        return edges
    ids = nodes.filter(pl.col("node_type") == NodeType.PROTEIN.value).select("node_id")
    new = (pe.select(
                (pl.lit("protein:") + pl.col("src")).alias("src"),
                (pl.lit("protein:") + pl.col("dst")).alias("dst"),
                pl.lit(EdgeType.PROTEIN_EXACT.value).alias("edge_type"),
                pl.struct(["pident", "alnlen"]).map_elements(
                    lambda r: _json.dumps({"pident": float(r["pident"]),
                                           "alnlen": int(r["alnlen"])}, sort_keys=True),
                    return_dtype=pl.Utf8).alias("props"))
             .join(ids.rename({"node_id": "src"}), on="src", how="semi")
             .join(ids.rename({"node_id": "dst"}), on="dst", how="semi"))
    log.info("protein_exact: %d edges (%d pairs dropped — endpoint not a Protein node)",
             new.height, pe.height - new.height)
    return pl.concat([edges, new], how="vertical_relaxed")


def _wire_reference_provenance(
    raw_edges: pl.DataFrame,
    canonical_edges: pl.DataFrame,
    nodes_for_wire: pl.DataFrame,
    processed: Path,
    log: logging.Logger,
) -> pl.DataFrame:
    """Synthesize Example -> Assay / Publication / Protein direct edges by
    collapsing multi-hop paths through the raw KG's ChEMBL Activity and
    BindingDB record subgraphs.

    The raw kg_edges parquet carries the full reference-DB provenance
    (chembl_activity_has_assay / _has_document / _has_target,
    bindingdb_ligand_in_publication / _targets_protein,
    bindingdb_record_has_ligand / _has_protein), but the canonical schema
    only encodes single-edge axis relationships. We compose direct edges
    here so the audit can traverse the assay / publication axes without
    expanding multi-hop paths every query.

    Returns the canonical edges with new synthetic edges concatenated.
    Caller is responsible for downstream dedup.
    """
    import json as _json
    # All synthesis chains start with Example -> benchmark Ligand.
    ex_lig = (raw_edges.filter(pl.col("edge_type") == "example_has_ligand")
              .select([pl.col("src").alias("example_id"),
                       pl.col("dst").alias("bench_lid")])
              .unique())
    n_synth_pub = n_synth_assay = n_synth_prot = 0

    # ---- ChEMBL chain ----
    # Build the chain in *aggregated* steps so we never materialise the full
    # Cartesian explosion. Each pair set is deduped before the next join.
    bench_to_chembl = (raw_edges.filter(
            pl.col("edge_type") == "benchmark_ligand_same_inchikey_as_chembl_ligand")
        .select([pl.col("src").alias("bench_lid"),
                 pl.col("dst").alias("chembl_lid")])
        .unique())
    if bench_to_chembl.height:
        chembl_lig_to_act = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_ligand")
            .select([pl.col("dst").alias("chembl_lid"),
                     pl.col("src").alias("chembl_act_id")])
            .unique())
        log.info("  chembl chain: %d bench->chembl_lig, %d chembl_lig->act",
                 bench_to_chembl.height, chembl_lig_to_act.height)

        # Activity -> Document, deduped first to drop redundant pairs.
        act_doc = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_document")
            .select([pl.col("src").alias("chembl_act_id"),
                     pl.col("dst").alias("chembl_doc_id")])
            .unique())
        if act_doc.height:
            chembl_lig_to_doc = (chembl_lig_to_act
                                 .join(act_doc, on="chembl_act_id", how="inner")
                                 .select(["chembl_lid", "chembl_doc_id"]).unique())
            bench_to_doc = (bench_to_chembl
                            .join(chembl_lig_to_doc, on="chembl_lid", how="inner")
                            .select(["bench_lid", "chembl_doc_id"]).unique())
            log.info("  chembl: %d unique chembl_lig->doc, %d bench->doc",
                     chembl_lig_to_doc.height, bench_to_doc.height)
            ex_to_doc = (ex_lig
                         .join(bench_to_doc, on="bench_lid", how="inner")
                         .select([pl.col("example_id").alias("src"),
                                  pl.col("chembl_doc_id").alias("dst")])
                         .unique())
            ex_to_doc = ex_to_doc.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_PUBLICATION.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "ChEMBL"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_doc],
                                        how="vertical_relaxed")
            n_synth_pub += ex_to_doc.height

        # Activity -> Assay, same aggregated pattern. The full join produces
        # ~57M Example->Assay pairs which is enough to OOM the consolidator.
        # NO CAP. There used to be a cap of 5 assays per benchmark Ligand, put here
        # to bound memory — not for any scientific reason. It silently truncated
        # 71 % of examples, and it truncated them with a systematic bias: the kept
        # assays were the 5 with the smallest ChEMBL id, i.e. the OLDEST. Two
        # ligands sharing a recent assay were simply never linked, and no
        # downstream step could recover an edge the KG never recorded. The KG is
        # supposed to state what is true; deciding which assays are too promiscuous
        # to count is a downstream policy, and `degree` is recorded so it can.
        act_asy = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_assay")
            .select([pl.col("src").alias("chembl_act_id"),
                     pl.col("dst").alias("chembl_asy_id")])
            .unique())
        if act_asy.height:
            chembl_lig_to_asy = (chembl_lig_to_act
                                 .join(act_asy, on="chembl_act_id", how="inner")
                                 .select(["chembl_lid", "chembl_asy_id"]).unique())
            bench_to_asy = (bench_to_chembl
                            .join(chembl_lig_to_asy, on="chembl_lid", how="inner")
                            .select(["bench_lid", "chembl_asy_id"]).unique())
            bench_to_asy = bench_to_asy.sort(["bench_lid", "chembl_asy_id"])
            log.info("  chembl: %d unique chembl_lig->asy, %d bench->asy (NO CAP)",
                     chembl_lig_to_asy.height, bench_to_asy.height)
            ex_to_asy = (ex_lig
                         .join(bench_to_asy, on="bench_lid", how="inner")
                         .select([pl.col("example_id").alias("src"),
                                  pl.col("chembl_asy_id").alias("dst")])
                         .unique())
            ex_to_asy = ex_to_asy.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_ASSAY.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "ChEMBL"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_asy],
                                        how="vertical_relaxed")
            n_synth_assay += ex_to_asy.height

        # ---- ChEMBL: ligand -> protein it was MEASURED against ----
        # `ligand_measured_protein` is documented as "real evidence of pretraining
        # contamination — a ChEMBL/BindingDB-trained model has seen that (ligand,
        # protein) pair". It was BindingDB-only: 378,427 pairs over 140,457 ligands.
        # ChEMBL — the corpus these models are actually pretrained on — contributed
        # NOTHING, because `chembl_activity_has_target` was never mapped to the
        # canonical schema and `chembl_targets.parquet` never carried an accession.
        # ChEMBL has 3,436,257 such pairs over 604,978 ligands to give.
        #
        # Only pairs whose protein ALREADY has a node are emitted (2,856,882 of them,
        # over 4,437 proteins). The other 579,375 would need 3,368 new Protein nodes —
        # and adding proteins changes the MMseqs clustering, which changes the protein
        # axis's leakage groups. That is a decision about the experiment, not a bug
        # fix, so it is left open and logged rather than taken silently.
        acc_f = processed / "chembl_target_accessions.parquet"
        act_tgt = (raw_edges.filter(
                pl.col("edge_type") == "chembl_activity_has_target")
            .select([pl.col("src").alias("chembl_act_id"),
                     pl.col("dst").alias("chembl_tgt_id")])
            .unique())
        if act_tgt.height and acc_f.exists():
            acc = (pl.read_parquet(acc_f)
                     .select((pl.lit("chembl_tgt:") + pl.col("target_chembl_id"))
                             .alias("chembl_tgt_id"),
                             (pl.lit("protein:") + pl.col("accession")).alias("dst"))
                     .unique())
            lig_to_prot_chembl = (
                bench_to_chembl
                .join(chembl_lig_to_act, on="chembl_lid", how="inner")
                .join(act_tgt, on="chembl_act_id", how="inner")
                .select(["bench_lid", "chembl_tgt_id"]).unique()
                .join(acc, on="chembl_tgt_id", how="inner")
                .select([pl.col("bench_lid").alias("src"), "dst"]).unique())
            n_all = lig_to_prot_chembl.height
            prot_ids = nodes_for_wire.filter(
                pl.col("node_type") == NodeType.PROTEIN.value).select("node_id")
            lig_to_prot_chembl = lig_to_prot_chembl.join(
                prot_ids.rename({"node_id": "dst"}), on="dst", how="semi")
            log.info("  chembl: %d (ligand, protein) measured pairs, %d kept "
                     "(%d dropped — protein has no node yet)",
                     n_all, lig_to_prot_chembl.height,
                     n_all - lig_to_prot_chembl.height)
            if lig_to_prot_chembl.height:
                lig_to_prot_chembl = lig_to_prot_chembl.with_columns([
                    pl.lit(EdgeType.LIGAND_MEASURED_PROTEIN.value).alias("edge_type"),
                    pl.lit(_json.dumps({"source": "ChEMBL"})).alias("props"),
                ])
                canonical_edges = pl.concat([canonical_edges, lig_to_prot_chembl],
                                            how="vertical_relaxed")
                n_synth_prot += lig_to_prot_chembl.height

    # ---- BindingDB chain ----
    bench_to_bdb = (raw_edges.filter(
            pl.col("edge_type") == "benchmark_ligand_same_inchikey_as_bindingdb_ligand")
        .select([pl.col("src").alias("bench_lid"),
                 pl.col("dst").alias("bdb_lid")])
        .unique())
    if bench_to_bdb.height:
        # bdb_lig -> Publication
        bdb_pub = (raw_edges.filter(
                pl.col("edge_type") == "bindingdb_ligand_in_publication")
            .select([pl.col("src").alias("bdb_lid"),
                     pl.col("dst").alias("pub_id")])
            .unique())
        if bdb_pub.height:
            ex_to_bdb_pub = (ex_lig
                             .join(bench_to_bdb, on="bench_lid", how="inner")
                             .join(bdb_pub, on="bdb_lid", how="inner")
                             .select([pl.col("example_id").alias("src"),
                                      pl.col("pub_id").alias("dst")])
                             .unique())
            ex_to_bdb_pub = ex_to_bdb_pub.with_columns([
                pl.lit(EdgeType.EXAMPLE_FROM_PUBLICATION.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "BindingDB"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, ex_to_bdb_pub],
                                        how="vertical_relaxed")
            n_synth_pub += ex_to_bdb_pub.height

        # bdb_lig -> Protein (UniProt), emitted as Ligand -> Protein directly.
        #
        # This used to be emitted as `example_has_protein`, which then had to be
        # untangled again by `split_example_protein_relation`. Two things went
        # wrong in that round trip:
        #
        #  - The Example -> Protein edge from BindingDB collided with the corpus
        #    loader's own target edge whenever a ligand had been measured against
        #    its own target. Dedup on (src, dst) kept ONE of them, and it kept the
        #    BindingDB row: 305,668 target edges ended up claiming
        #    props.source = "BindingDB" — a false statement about where the fact
        #    came from.
        #  - Those same collisions were then EXCLUDED from ligand_measured_protein
        #    (the splitter only routes `dst != true_prot` there), so "this ligand
        #    was measured against this very target" — the strongest evidence of
        #    pretraining contamination there is — was the one case the graph did
        #    not record.
        #
        # The relation is Ligand -> Protein. Emit it as Ligand -> Protein.
        bdb_prot = (raw_edges.filter(
                pl.col("edge_type") == "bindingdb_ligand_targets_protein")
            .select([pl.col("src").alias("bdb_lid"),
                     pl.col("dst").alias("prot_id")])
            .unique())
        if bdb_prot.height:
            lig_to_prot = (ex_lig.select("bench_lid").unique()
                           .join(bench_to_bdb, on="bench_lid", how="inner")
                           .join(bdb_prot, on="bdb_lid", how="inner")
                           .select([pl.col("bench_lid").alias("src"),
                                    pl.col("prot_id").alias("dst")])
                           .unique())
            lig_to_prot = lig_to_prot.with_columns([
                pl.lit(EdgeType.LIGAND_MEASURED_PROTEIN.value).alias("edge_type"),
                pl.lit(_json.dumps({"source": "BindingDB"})).alias("props"),
            ])
            canonical_edges = pl.concat([canonical_edges, lig_to_prot],
                                        how="vertical_relaxed")
            n_synth_prot += lig_to_prot.height

        # BindingDB record-as-Assay was attempted but produces a 785K
        # bdb_rec node set of which ~66 % (518K) are orphan after wiring —
        # they're activities on BindingDB ligands that no benchmark Example
        # references. The remaining signal duplicates the ChEMBL Assay axis
        # (BindingDB record's source paper is in BindingDB's Publication
        # axis already), so we skip this edge type to keep the canonical
        # KG focused.

    log.info("wire_ref_provenance: synth %d publication + %d assay + %d protein edges",
             n_synth_pub, n_synth_assay, n_synth_prot)
    return canonical_edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def consolidate(
    output_dir: Path,
    *,
    corpus: str = "all",
    limit: int | None = None,
    hub_cfg: HubMitigationConfig | None = None,
) -> BuildStats:
    """Consolidate the raw KG into the canonical schema parquets.

    Parameters
    ----------
    output_dir
        Where to write canonical_nodes.parquet, canonical_edges.parquet,
        stats.csv.
    corpus
        "all" -> read the merged kg_nodes/kg_edges from `build_kg`.
        Otherwise the per-corpus parquet name ("litpcba_ave", "dude",
        "dekois", "bigbind", "bayesbind").
    limit
        Optional row cap for smoke-testing.
    hub_cfg
        Hub-mitigation parameters; defaults to schema.HubMitigationConfig.
    """
    cfg = hub_cfg or HubMitigationConfig()
    processed = processed_dir()
    require_data_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_name = "kg_nodes" if corpus == "all" else f"{corpus}_nodes"
    edges_name = "kg_edges" if corpus == "all" else f"{corpus}_edges"

    nodes_path = processed / f"{nodes_name}.parquet"
    edges_path = processed / f"{edges_name}.parquet"
    for p in (nodes_path, edges_path):
        if not p.exists():
            raise FileNotFoundError(p)

    stats = BuildStats()
    t0 = time.perf_counter()

    # Read EAGERLY so we only hit NFS once per file - critical on slow/loaded
    # NFS storage where mmap'd scan_parquet causes many small page faults.
    nodes = pl.read_parquet(nodes_path)
    edges = pl.read_parquet(edges_path)
    raw_edges_full = edges       # preserved for reference-provenance wiring below
    if limit:
        nodes = nodes.head(limit)
        edges = edges.head(limit)
        raw_edges_full = raw_edges_full.head(limit)

    stats.n_nodes_in = nodes.height
    stats.n_edges_in = edges.height
    log.info("read %d rows from %s", stats.n_nodes_in, nodes_path.name)
    log.info("read %d rows from %s", stats.n_edges_in, edges_path.name)

    nodes, dropped_n = _map_nodes(nodes)
    edges, dropped_e = _map_edges(edges)
    stats.n_nodes_dropped = dropped_n
    stats.n_edges_dropped = dropped_e
    log.info("mapped: n_nodes=%d (-%d), n_edges=%d (-%d)",
             nodes.height, dropped_n, edges.height, dropped_e)

    # Collapse `tgt:Corpus:UniProtID` Protein nodes onto canonical
    # `protein:UniProtID` so the protein axis isn't split across two synonyms.
    nodes, edges, n_protein_collapsed = _normalize_protein_ids(nodes, edges)
    if n_protein_collapsed:
        log.info("collapsed %d tgt:Corpus:UniProt Protein nodes onto protein:UniProt",
                 n_protein_collapsed)
        stats.deferred = (stats.deferred or []) + [
            f"protein_id_collapsed={n_protein_collapsed}"
        ]

    # Defect 1: map gene-symbol targets onto their UniProt accession BEFORE the
    # cluster edges are attached, otherwise they attach to nothing.
    nodes, edges, n_mapped = _fixes.apply_protein_id_map(nodes, edges, processed)
    log.info("protein id map: rewrote %d nodes", n_mapped)

    nodes, edges = _add_protein_cluster_edges(edges, nodes, processed, stats)
    log.info("after cluster edges: n_nodes=%d, n_edges=%d", nodes.height, edges.height)

    # Protein <-> Protein identity. Declared in the schema at weight 1.00, in the
    # protein axis, and never once emitted — see _add_protein_exact_edges.
    edges = _add_protein_exact_edges(nodes, edges, processed, stats)


    kg_out = Path(output_dir) if output_dir else processed
    nodes, edges, n_stereo = _fixes.merge_stereo_scaffolds(nodes, edges, kg_out)
    log.info("merged %d stereo-duplicate Scaffold nodes", n_stereo)

    edges, n_multi_sc = _fixes.one_scaffold_per_ligand(edges)
    log.info("dropped %d surplus ligand_scaffold edges (1 scaffold per ligand)", n_multi_sc)

    nodes, edges, n_trivial = _drop_trivial_scaffolds(nodes, edges, cfg)
    stats.n_trivial_scaffolds_dropped = n_trivial
    log.info("annotated %d Scaffold nodes with n_heavy_atoms", n_trivial or nodes.filter(pl.col("node_type") == NodeType.SCAFFOLD.value).height)

    edges, n_relabel = _fixes.relabel_identical_similars(edges)
    log.info("relabelled %d ligand_similar edges at T>=0.9995 -> fingerprint_exact", n_relabel)

    edges, n_sorted = _fixes.sort_ligand_pairs(edges)
    log.info("sorted %d unordered ligand_exact/parent_exact pairs", n_sorted)

    nodes, edges, n_noligand = _fixes.drop_ligandless_examples(nodes, edges)
    log.info("dropped %d Examples with no Ligand", n_noligand)

    nodes, edges, proto_counts = _fixes.rebuild_decoy_protocol(nodes, edges)
    log.info("decoy-protocol axis rebuilt: %s", proto_counts)


    # Prune dangling edges (cluster edges typically dangle when corpus-level
    # Protein ids don't match the UniProt-based cluster member ids). Done
    # BEFORE wiring so the prune only walks the ~19.7M corpus-level edges,
    # not the ~86M post-wire set.
    _ids = nodes.select("node_id")
    n_before = edges.height
    edges = (edges.join(_ids.rename({"node_id": "src"}), on="src", how="semi")
                  .join(_ids.rename({"node_id": "dst"}), on="dst", how="semi"))
    pruned = n_before - edges.height
    if pruned:
        log.info("pruned %d dangling edges", pruned)
        stats.deferred = (stats.deferred or []) + [f"pruned_dangling_edges={pruned}"]

    # B1+B2+B3: synthesize Example -> Publication / Assay / Protein direct
    # edges by collapsing multi-hop ChEMBL + BindingDB provenance chains.
    # The wire emits edges from Example ids to canonical Doc/Assay/Protein
    # ids that survived `_map_nodes`, so they don't dangle by construction
    # and we skip a second prune pass.
    n_before_wire = edges.height
    edges = _wire_reference_provenance(raw_edges_full, edges, nodes, processed, log)
    log.info("after wire: n_edges=%d (+%d new)", edges.height, edges.height - n_before_wire)


    # wire re-emits protein edges from the raw frame, which reintroduces the
    # malformed BindingDB protein ids — re-apply the map to catch them.
    nodes, edges, n_remap2 = _fixes.apply_protein_id_map(nodes, edges, processed)
    log.info("protein id map (post-wire): %d", n_remap2)

    # example_has_protein carried two relations; separate them (see fixes.py).
    edges, n_split, n_measured = _fixes.split_example_protein_relation(nodes, edges, processed)
    log.info("split example_has_protein: %d non-target edges -> %d ligand_measured_protein",
             n_split, n_measured)

    # ChEMBL doc_type=DATASET is a placeholder, not a publication.
    #
    # This path used to be the string literal
    # "data/raw/ChEMBL/extracted/chembl_35/chembl_35_sqlite/chembl_35.db" — RELATIVE to
    # the current working directory. Run consolidate from anywhere but the repo root and
    # the database is not found, `drop_placeholder_publications` logs a warning and
    # returns unchanged, and the KG ships with `CHEMBL1201862` ("PubChem BioAssay data
    # set") joining 1,730,288 Examples — 34 % of the graph — through a single
    # Publication node. The publication axis collapses to one group covering 95 % of the
    # corpus, and the only sign is one warning line in a log nobody reads.
    # The module already has `raw_dir()`. Use it.
    chembl_db = (require_data_root() / "data" / "raw" / "ChEMBL" / "extracted"
                 / "chembl_35" / "chembl_35_sqlite" / "chembl_35.db")
    if not chembl_db.exists():
        raise FileNotFoundError(
            f"ChEMBL DB not found at {chembl_db}. Placeholder DATASET documents cannot "
            f"be filtered, and leaving them in collapses the publication axis into one "
            f"group covering 95 % of the corpus. Refusing to build a KG that would be "
            f"quietly wrong.")
    edges, n_ph = _fixes.drop_placeholder_publications(nodes, edges, chembl_db)
    log.info("dropped %d publication edges to placeholder DATASET docs", n_ph)

    # Universal orphan drop: any node with degree 0 after the dangling-edge
    # prune is removed. This covers Protein/ProteinCluster (cluster member ids
    # that don't match KG protein ids), Assay/Publication (when reference-DB
    # wiring fails to reach them), and any other isolated node accumulated by
    # the build. The five small-cardinality "structure" types (DatasetSource,
    # DecoyProtocol, plus everything Example-side) are excluded so a corpus
    # with zero edges still keeps its DatasetSource pin.
    touched_ids = pl.concat(
        [edges.select(pl.col("src").alias("node_id")),
         edges.select(pl.col("dst").alias("node_id"))],
        how="vertical_relaxed",
    ).unique()
    keep_anyway = {"DatasetSource", "DecoyProtocol"}
    keep_pinned = nodes.filter(pl.col("node_type").is_in(list(keep_anyway)))
    drop_candidates = nodes.filter(~pl.col("node_type").is_in(list(keep_anyway)))
    kept_via_edges = drop_candidates.join(touched_ids, on="node_id", how="semi")
    n_before_n = nodes.height
    nodes = pl.concat([keep_pinned, kept_via_edges], how="vertical_relaxed")
    dropped_orphans = n_before_n - nodes.height
    if dropped_orphans:
        by_type = (drop_candidates
                   .join(touched_ids, on="node_id", how="anti")
                   .group_by("node_type").len()
                   .sort("len", descending=True))
        log.info("dropped %d orphan nodes: %s", dropped_orphans,
                 dict(by_type.iter_rows()))
        stats.deferred = (stats.deferred or []) + [f"dropped_orphans={dropped_orphans}"]

    # Targeted dedup (a global `unique` on 38M+ edges OOMs the 22 GB box):
    #   - `ligand_similar`: per-corpus loader emits `ligand_similar_to_ligand`
    #     which collides with D5's `ligand_similar` after canonical mapping,
    #     and the upstream sim job doesn't enforce src<dst on the unordered
    #     pair, so 120 pairs land in both directions. Force sorted-pair, then
    #     dedup.
    #   - `example_has_protein`: protein-id normalisation can collapse a
    #     corpus `tgt:BigBind:X` Protein onto the canonical `protein:X`,
    #     leaving a duplicate Example -> protein:X edge if both the corpus
    #     loader and the BindingDB wire pointed at the same UniProt.
    # Every other edge type has globally unique (src, dst) by construction.
    ls = edges.filter(pl.col("edge_type") == "ligand_similar")
    if ls.height:
        n_before_e = ls.height
        ls = ls.with_columns([
            pl.min_horizontal("src", "dst").alias("_a"),
            pl.max_horizontal("src", "dst").alias("_b"),
        ]).with_columns([
            pl.col("_a").alias("src"),
            pl.col("_b").alias("dst"),
        ]).drop(["_a", "_b"]).pipe(_fixes.stable_unique, ["src", "dst"])
        deduped = n_before_e - ls.height
        edges = pl.concat([
            edges.filter(pl.col("edge_type") != "ligand_similar"),
            ls,
        ], how="vertical_relaxed")
        if deduped:
            log.info("deduped %d redundant ligand_similar edges (incl. bidir)", deduped)
            stats.deferred = (stats.deferred or []) + [f"deduped_ligand_similar={deduped}"]

    #   - `ligand_measured_protein`: a (ligand, protein) pair can be measured in BOTH
    #     ChEMBL and BindingDB, and each chain emits it. That is one fact, not two.
    #     Record both provenances in props rather than shipping a duplicate triple.
    lmp = edges.filter(pl.col("edge_type") == "ligand_measured_protein")
    if lmp.height:
        n_before_m = lmp.height
        lmp = (lmp.with_columns(
                   pl.col("props").str.json_path_match("$.source").alias("_s"))
                  .group_by("src", "dst")
                  .agg(pl.col("_s").drop_nulls().unique().sort().alias("_srcs"))
                  .with_columns(
                      pl.lit(EdgeType.LIGAND_MEASURED_PROTEIN.value).alias("edge_type"),
                      pl.col("_srcs").list.join("+").alias("_j"))
                  .with_columns(
                      (pl.lit('{"source": "') + pl.col("_j") + pl.lit('"}')).alias("props"))
                  .select("src", "dst", "edge_type", "props"))
        deduped_m = n_before_m - lmp.height
        edges = pl.concat([
            edges.filter(pl.col("edge_type") != "ligand_measured_protein"),
            lmp,
        ], how="vertical_relaxed")
        if deduped_m:
            log.info("merged %d ligand_measured_protein edges measured in both sources",
                     deduped_m)

    ehp = edges.filter(pl.col("edge_type") == "example_has_protein")
    if ehp.height:
        n_before_p = ehp.height
        ehp = ehp.pipe(_fixes.stable_unique, ["src", "dst"])
        deduped_p = n_before_p - ehp.height
        edges = pl.concat([
            edges.filter(pl.col("edge_type") != "example_has_protein"),
            ehp,
        ], how="vertical_relaxed")
        if deduped_p:
            log.info("deduped %d redundant example_has_protein edges (post protein-id collapse)",
                     deduped_p)
            stats.deferred = (stats.deferred or []) + [f"deduped_example_has_protein={deduped_p}"]

    # label_type + split onto Example props. Before degree, after every edge fix:
    # it reads edges, so it must see the final edge set.
    nodes = _enrich_example_props(nodes, edges)

    # Degree LAST — after every edge-removing step above.
    #
    # This used to run before the two dedup passes, so the duplicate edges they
    # remove were still being counted. Measured on the shipped KG: 306,704 nodes
    # carried a degree higher than their true edge count (305,668 Examples off by
    # one, 1,036 Proteins off by up to 6,301; 611,336 phantom edges in total).
    # `degree` is the fact that replaced the `is_hub` policy — it is what the
    # downstream weight-discount reads. A stale count is not a smaller sin than a
    # frozen threshold; it is the same sin with a decimal point.
    nodes, edges, n_hubs = _shard_hub_nodes(nodes, edges, cfg)
    stats.n_hub_nodes_sharded = n_hubs
    log.info("annotated node degree (max=%d)", int(nodes["degree"].max()))

    # Already eager DataFrames at this point.
    nodes_df = nodes
    edges_df = edges
    stats.n_nodes_out = nodes_df.height
    stats.n_edges_out = edges_df.height
    stats.nodes_by_type = dict(
        nodes_df.group_by("node_type").len().sort("len", descending=True).iter_rows()
    )
    stats.edges_by_type = dict(
        edges_df.group_by("edge_type").len().sort("len", descending=True).iter_rows()
    )
    # Things we cannot compute on this box without an encoder.
    stats.deferred = (stats.deferred or []) + [
        "time_overlap_edges_need_ChEMBL_dates",
        "example_from_assay_needs_chembl_assay_join",
        "example_from_publication_needs_chembl_document_join",
    ]

    _fixes.validate_canonical(nodes_df, edges_df)
    log.info("write-time validation passed")

    # Deterministic row order. Without it two runs of the same code over the same
    # input produce the same GRAPH but different bytes, so a rebuild cannot be
    # checksum-verified against the shipped KG — which is exactly the check that
    # would have caught the intermittent polars corruption earlier.
    nodes_df = nodes_df.sort("node_id")
    edges_df = edges_df.sort(["edge_type", "src", "dst"])

    nodes_out = output_dir / "canonical_nodes.parquet"
    edges_out = output_dir / "canonical_edges.parquet"
    stats_out = output_dir / "stats.csv"

    nodes_df.write_parquet(nodes_out)
    edges_df.write_parquet(edges_out)
    pl.DataFrame(stats.to_csv_rows()).write_csv(stats_out)

    log.info(
        "canonical KG: %s nodes, %s edges, wrote to %s (%.1fs)",
        stats.n_nodes_out,
        stats.n_edges_out,
        output_dir,
        time.perf_counter() - t0,
    )
    return stats


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--corpus", default="all",
                   choices=["all", "litpcba_ave", "dude", "dekois", "bigbind", "bayesbind"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stats = consolidate(
        output_dir=args.output_dir,
        corpus=args.corpus,
        limit=args.limit,
    )
    print(f"nodes_out={stats.n_nodes_out} edges_out={stats.n_edges_out}")


if __name__ == "__main__":
    _cli()
