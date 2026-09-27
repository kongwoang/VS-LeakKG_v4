"""Forbidden-relation regimes (proposal §3.4, Table 2, and the release list in §3.6).

A regime names a set of AXES; its forbidden edge set F is the union of
schema.AXIS_EDGE_TYPES over those axes. Two examples connected in the subgraph
kept by F may not straddle train/val/test.

F is deliberately built ONLY from schema.AXIS_EDGE_TYPES, so it can never contain a
NON_AXIS edge type (ligand_measured_protein, example_in_split, example_has_label_type,
example_in_trainset). Those are hubs by construction and would merge the corpus into
one group — see schema.EdgeType and docs/KG_CONSTRUCTION.md.

Pocket-clean from proposal §3.6 is intentionally absent: the pocket axis was removed
in the v3 redesign and never rebuilt (see CONTEXT.md §3). Adding it here without the
axis would silently produce a ligand-or-protein split mislabelled "pocket-clean".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vsleakkg.kg import schema


# Axes whose grouping is driven by a promiscuous hub node (a common scaffold, a
# distant-family cluster, a high-throughput assay, a popular year). For these a
# clean split MUST choose a degree cap or the largest group swallows the corpus
# (audit_semantics reports the curve). Ligand/source are hub-free at identity.
HUB_PRONE_AXES: frozenset[str] = frozenset({"scaffold", "protein", "assay", "time"})


@dataclass(frozen=True)
class Regime:
    name: str
    axes: tuple[str, ...]
    # Policy knobs, both reported in the split manifest, never hidden.
    tanimoto_floor: float = 0.80          # which ligand_similar edges count
    degree_cap: int | None = 1000         # drop content nodes with degree > cap; None = keep all
    # Provenance axes (assay/publication/time) have coverage confounded with the
    # label (CONTEXT.md §3): a decoy has no assay/pub/year, so a group built on them
    # is systematically an actives-only group. Flagged so the splitter and any
    # decile analysis stratify by label instead of reading the confound as signal.
    label_confounded: bool = False

    @property
    def forbidden_edge_types(self) -> frozenset[str]:
        f: set[str] = set()
        for ax in self.axes:
            f.update(schema.AXIS_EDGE_TYPES[ax])
        # Belt and braces: F is a subset of the union of axes, so this can only be
        # empty, never a NON_AXIS type. Assert it anyway — a future edit to
        # AXIS_EDGE_TYPES that leaked a hub type in would be caught here.
        assert not (f & schema.NON_AXIS_EDGE_TYPES), \
            f"regime {self.name} would traverse non-axis edges: {f & schema.NON_AXIS_EDGE_TYPES}"
        return frozenset(f)

    @property
    def anchor_edge_types(self) -> frozenset[str]:
        """example_* edges that attach examples to content nodes, per axis."""
        return frozenset(t for t in self.forbidden_edge_types if t.startswith("example_"))


REGIMES: dict[str, Regime] = {
    r.name: r
    for r in [
        Regime("ligand-clean", ("ligand",), degree_cap=None),
        Regime("scaffold-clean", ("scaffold",)),
        Regime("protein-clean", ("protein",), degree_cap=None),
        Regime("assay-clean", ("assay",), label_confounded=True),
        Regime("dual-clean", ("ligand", "scaffold", "protein")),
        Regime(
            "strict-clean",
            ("ligand", "scaffold", "protein", "assay", "source", "time"),
            label_confounded=True,
        ),
    ]
}
