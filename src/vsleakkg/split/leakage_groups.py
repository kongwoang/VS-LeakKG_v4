"""Leakage-group construction (proposal §3.5).

Two benchmark examples are in the same leakage group iff they are connected in the
subgraph that keeps only the regime's forbidden edge types F (the example_* anchors
plus the content-content relations F names). Every example in a group must land in
one partition (§3.6); groups are the atoms the splitter assigns.

This is the exact, deterministic half of the downstream layer: connected components,
no weights, no policy — except the two knobs a clean split cannot avoid, both surfaced
in the returned stats, never hidden:

  - tanimoto_floor : which ligand_similar edges count. The KG stores every edge with
                     Tanimoto >= 0.80 and keeps the value in props; the split floor is
                     a downstream choice, so it is applied here, not in the graph.
  - degree_cap     : drop a content node (scaffold, cluster, assay, publication,
                     timebin) once its degree exceeds the cap, so one benzene / one
                     popular year cannot collapse the corpus into a single group.
                     This is proposal §5.3 hub mitigation; audit_semantics reports the
                     full block-size-vs-cap curve. None = keep every node.

Correctness notes carried over from the audit_semantics prototype:
  * Edges are filtered as PAIRS. Filtering src and dst independently and zipping the
    survivors (as an earlier version did) fabricates edges between unrelated nodes and
    silently fuses components. See the comment in tools/audit_semantics.py.
  * F is validated against schema.NON_AXIS_EDGE_TYPES: ligand_measured_protein /
    example_in_split / example_has_label_type are hubs by construction and are refused.
  * Examples are never dropped by the degree cap — only content nodes are. Every one of
    the ~5.0M examples appears in the output; those touched by no forbidden relation are
    singleton groups.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vsleakkg.kg import schema
from vsleakkg.split.regimes import Regime


@dataclass(frozen=True)
class GroupResult:
    assignment: pl.DataFrame          # columns: example_id (str), group_id (int)
    n_examples: int
    n_groups: int
    largest_group: int                # example count in the biggest group
    regime: str
    tanimoto_floor: float
    degree_cap: int | None
    n_hub_nodes_dropped: int          # content nodes removed by the degree cap

    @property
    def largest_group_frac(self) -> float:
        return self.largest_group / self.n_examples if self.n_examples else 0.0

    def summary(self) -> str:
        cap = "none" if self.degree_cap is None else f"{self.degree_cap:,}"
        return (
            f"{self.regime:<14} T>={self.tanimoto_floor:.2f} cap={cap:<7} "
            f"{self.n_groups:>9,} groups | largest {self.largest_group:>9,} "
            f"= {100 * self.largest_group_frac:5.2f}% of corpus | "
            f"hub nodes dropped {self.n_hub_nodes_dropped:,}"
        )


def _forbidden_edges(
    edges: pl.LazyFrame,
    regime: Regime,
    degree: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Return (src, dst) pairs kept for the regime, and the count of hub nodes dropped.

    `degree` is a DataFrame[node_id, degree] for every node. The cap drops any
    content node (non-Example endpoint) whose degree exceeds regime.degree_cap.
    """
    f = list(regime.forbidden_edge_types)
    sub = edges.filter(pl.col("edge_type").is_in(f))

    # Apply the Tanimoto floor to ligand_similar only; every other forbidden edge
    # type is an identity/cluster relation with no threshold to sweep.
    if "ligand_similar" in regime.forbidden_edge_types and regime.tanimoto_floor > 0.80:
        keep_sim = (
            pl.col("edge_type") != "ligand_similar"
        ) | (
            pl.col("props").str.json_path_match("$.tanimoto").cast(pl.Float64)
            >= regime.tanimoto_floor
        )
        sub = sub.filter(keep_sim)

    pairs = sub.select("src", "dst").collect()

    n_hub_dropped = 0
    if regime.degree_cap is not None:
        deg = dict(zip(degree["node_id"], degree["degree"]))
        cap = regime.degree_cap
        # A node is "hub" if it is NOT an example and its degree exceeds the cap.
        # Examples are never dropped.
        def is_hub(node: str) -> bool:
            return not node.startswith("ex:") and deg.get(node, 0) > cap

        hub_nodes = {
            v
            for v in set(pairs["src"].to_list()) | set(pairs["dst"].to_list())
            if is_hub(v)
        }
        n_hub_dropped = len(hub_nodes)
        if hub_nodes:
            pairs = pairs.filter(
                ~pl.col("src").is_in(hub_nodes) & ~pl.col("dst").is_in(hub_nodes)
            )
    return pairs, n_hub_dropped


def build_groups(
    nodes: pl.LazyFrame,
    edges: pl.LazyFrame,
    regime: Regime,
) -> GroupResult:
    """Assign every Example to a leakage group under `regime`.

    nodes / edges are lazy scans of canonical_{nodes,edges}.parquet.
    Deterministic: node ids are sorted before indexing, so group ids depend only on
    the graph and the regime, not on row order or run.
    """
    examples = (
        nodes.filter(pl.col("node_type") == "Example")
        .select(pl.col("node_id").alias("id"))
        .collect()
    )
    degree = nodes.select("node_id", "degree").collect()

    pairs, n_hub_dropped = _forbidden_edges(edges, regime, degree)

    # Node universe = every example (so isolated ones survive as singletons) plus
    # every endpoint of a kept edge. Sorted for determinism.
    universe = (
        pl.concat([examples["id"], pairs["src"], pairs["dst"]])
        .unique()
        .sort()
    )
    idx = pl.DataFrame({"id": universe}).with_row_index("i")

    # Map endpoints to integer indices via joins (no Python-level dict over millions).
    e2 = (
        pairs.join(idx.rename({"id": "src", "i": "si"}), on="src")
        .join(idx.rename({"id": "dst", "i": "di"}), on="dst")
    )
    si = e2["si"].to_numpy()
    di = e2["di"].to_numpy()

    n_nodes = idx.height
    if len(si):
        g = coo_matrix(
            (np.ones(len(si), np.int8), (si, di)), shape=(n_nodes, n_nodes)
        )
        _, labels = connected_components(g, directed=False, return_labels=True)
    else:
        labels = np.arange(n_nodes, dtype=np.int64)

    ex_i = examples.join(idx, on="id")["i"].to_numpy()
    ex_labels = labels[ex_i]

    # Compact the component labels to contiguous group ids [0, n_groups).
    uniq, inv = np.unique(ex_labels, return_inverse=True)
    assignment = examples.rename({"id": "example_id"}).with_columns(
        pl.Series("group_id", inv.astype(np.int64))
    )

    _, counts = np.unique(inv, return_counts=True)
    return GroupResult(
        assignment=assignment,
        n_examples=examples.height,
        n_groups=len(uniq),
        largest_group=int(counts.max()) if len(counts) else 0,
        regime=regime.name,
        tanimoto_floor=regime.tanimoto_floor,
        degree_cap=regime.degree_cap,
        n_hub_nodes_dropped=n_hub_dropped,
    )
