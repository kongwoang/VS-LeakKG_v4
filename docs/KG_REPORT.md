# KG report

What the finished graph contains, how it was verified, and what it already says.
For *how* it is built and *why* each decision was made, see `KG_CONSTRUCTION.md`.

Built 2026-07-14. Deterministic — a rebuild reproduces it exactly.

```
outputs/kg/canonical_nodes.parquet   [node_id, node_type, label, props, degree]
outputs/kg/canonical_edges.parquet   [src, dst, edge_type, props]
outputs/kg/stats.csv
```

## 1. Composition

**8,574,944 nodes · 75,369,818 edges** (476 MB).

| node type | count | |
|---|---:|---|
| Example | 5,025,493 | one labelled protein–ligand sample |
| Ligand | 2,013,247 | deduplicated by canonical SMILES |
| Assay | 857,115 | ChEMBL + BindingDB |
| Scaffold | 569,721 | Bemis-Murcko, stereo-free |
| Publication | 92,987 | ChEMBL docs + BindingDB PMID/DOI |
| ProteinCluster | 10,903 | MMseqs2 @ 30 / 50 / 90 %, nested |
| Protein | 5,466 | UniProt-anchored (+ 3 HIV domains) |
| DatasetSource | 7 | |
| DecoyProtocol | 3 | one per corpus that generates decoys |
| DecoyProtocolClass | 2 | the method that protocol implements |

| edge type | count | axis |
|---|---:|---|
| `example_from_assay` | 51,570,095 | assay |
| `example_from_source` | 5,025,493 | source |
| `example_has_protein` | 5,025,493 | protein — exactly one target per example |
| `example_has_ligand` | 5,025,493 | ligand |
| `example_from_publication` | 3,784,386 | assay |
| `ligand_scaffold` | 2,013,247 | scaffold — exactly one per ligand |
| `source_decoy_protocol` | 1,753,639 | source — inactives only |
| `ligand_similar` | 448,586 | ligand — Tanimoto ∈ [0.80, 0.9995) |
| `ligand_measured_protein` | 378,427 | **none** — pretraining evidence, see below |
| `ligand_fingerprint_exact` | 314,683 | ligand — Tanimoto ≥ 0.9995 |
| `ligand_parent_exact` | 6,939 | ligand — same salt-stripped InChIKey |
| `ligand_exact` | 6,939 | ligand — same full InChIKey |
| `protein_cluster_30/50/90` | 5,465 each | protein — weights 0.45 / 0.65 / 0.85 |
| `decoy_protocol_in_class` | 3 | source |

Node degree: median 4, p99 104, max 2,651,977. **Degree is recorded as a fact; the
graph makes no judgement about it.** Deciding that a 330,122-compound HTS assay is too
promiscuous to count as leakage is a downstream policy — see `KG_CONSTRUCTION.md` §5.

`ligand_measured_protein` belongs to **no leakage axis** on purpose. It says "this
ligand has a measured activity against this protein" — real evidence of *pretraining*
contamination, a different question from benchmark-split leakage.

## 2. Verification

Two suites, both clean:

```bash
PYTHONPATH=src python tools/audit_kg.py         # 0 failed checks
PYTHONPATH=src python tools/audit_semantics.py
```

**`audit_kg.py` — 8 structural + 14 semantic invariants, all pass.** No duplicate
ids, no dangling edges, no self-loops, no duplicate triples, no orphans; every edge
connects the declared node types; every edge type has a leakage weight; every Example
has exactly one ligand, one source and one **target** protein; every Ligand has at most
one scaffold; all four ligand–ligand pair types are stored sorted; every Example is
visible to the protein-family axis; no Scaffold is duplicated modulo stereochemistry;
decoy protocols link only inactives, and the class tier spans corpora.

**`audit_semantics.py` — biology, chemistry, usability.**

- 20/20 target → UniProt spot-checks correct (COX-1 vs COX-2, ESR1 vs ESR2, AKT1 vs
  AKT2, nNOS, mTOR, …).
- HIV protease / RT / integrase stay in **separate clusters at all three resolutions** —
  the domain split is not undone by clustering.
- Protein clusters **nest**: 0 clusters at 90 % straddle a looser cluster.
- Scaffold SMILES: 2 / 20,000 fail to re-parse — an RDKit round-trip artefact
  (aromatic carbanions like `[c-]`), present in the inherited graph too.
- `ligand_similar` Tanimoto ∈ [0.800, 0.991] — 0 edges outside the declared range.
- 324 duplicate (corpus, target, ligand) triples — exactly the inherited baseline.

## 3. What the graph already says

### Cross-corpus protein overlap — invisible in the inherited graph

**142 proteins are shared by two or more corpora.** `P37231` (PPARγ) appears in DUD-E,
DEKOIS *and* LIT-PCBA. `P07900` (HSP90-α) is touched by **all five**. `P04150`
(glucocorticoid receptor), `P00533` (EGFR), `P12821` (ACE) — DUD-E and DEKOIS both.

The inherited graph could not see any of this: `tgt:DUD-E:hs90a` and
`tgt:DEKOIS:hsp90` were **two different nodes** for the same protein. Train on DUD-E,
test on DEKOIS, and you are testing on targets the model has already seen — and the
graph could not tell you.

### The provenance axes predict the label

| corpus | assay: active | assay: decoy | gap | publication: active | publication: decoy | gap |
|---|---:|---:|---:|---:|---:|---:|
| **DUD-E** | 91.8 % | 0.9 % | **91 pp** | 94.5 % | 0.3 % | **94 pp** |
| **DEKOIS** | 56.1 % | 2.8 % | **53 pp** | 58.3 % | 1.4 % | **57 pp** |
| LIT-PCBA | 57.6 % | 57.8 % | 0 pp | 6.5 % | 2.8 % | 4 pp |
| BigBind | 99.5 % | 98.2 % | 1 pp | 91.1 % | 57.3 % | 34 pp |
| BayesBind | 96.5 % | 98.7 % | 2 pp | 34.6 % | 74.1 % | 40 pp |

In DUD-E, the question *"does this example have a publication edge?"* predicts the
label almost perfectly. The reason is mundane: DUD-E's actives come from ChEMBL, its
decoys come from ZINC and were never assayed or published.

This cuts both ways.

**It is a finding.** It is precisely the *source-only shortcut* of proposal §3.8: the
label is recoverable from provenance alone, with no protein–ligand modelling at all.

**It is a trap.** Any contamination score on the assay or publication axis will be
systematically higher for actives than for decoys, so a contamination-decile analysis
on those axes will be confounded with the label unless it is stratified.

## 4. Can each axis be partitioned?

Proposal §3.5 builds splits by collapsing examples joined by forbidden relations into
atomic **leakage groups** and partitioning the groups. A group that swallows most of
the corpus makes that axis unsplittable — so it must be measured, not assumed.

Largest atomic block, as a fraction of the **whole corpus** (5,025,493 examples).
`cut` excludes intermediate nodes above a degree; the graph does not choose a cut-off
for you, so here is the curve:

| axis | coverage | cut = none | 100 k | 10 k | 1 k | 100 |
|---|---:|---:|---:|---:|---:|---:|
| ligand | 100 % | **0.1 %** | | | | |
| scaffold | 100 % | 2.2 % | | 0.3 % | **0.1 %** | |
| protein @ 90 % | 100 % | **7.2 %** | | | | |
| protein @ 30 % | 100 % | **8.5 %** | | | | |
| assay | 48 % | 44.8 % | 36.8 % | 21.2 % | 13.5 % | **8.2 %** |
| publication | 48 % | 13.0 % | | 13.0 % | 12.4 % | **7.8 %** |
| time | **0 %** | — | | | | |

**Ligand, scaffold and protein split cleanly** with no cut-off at all.

**Assay needs a degree cut-off.** Uncapped it is dominated by a handful of monster HTS
screens (max 330,122 compounds); at `cut = 100` the largest block is 8.2 %.

**Publication is usable but constrains the split.** Its largest block is 13 % of the
corpus — placeable in train for an 80/20 split, but every example in it (the
well-published compounds) can then never be tested. That bias must be disclosed, not
hidden. Note the block is *not* caused by one hub: after removing the placeholder
ChEMBL `DATASET` documents, the largest publication node has only 12,719 examples. It
is a **chain** — popular compounds appear in many papers, and examples link through
each other.

**The time axis is empty.** It is declared in the schema and promised by the proposal;
it carries zero edges. It needs ChEMBL document dates.

## 5. Known limits

- **Pocket axis: absent.** Removed in the v3 redesign. `data/raw/DUD-E_pockets_fetched/`
  is preserved if it is revived.
- **Time axis: empty.**
- **`example_from_assay` / `example_from_publication` are ligand-mediated** — "this
  example's ligand was tested in / reported in", not "this example's label came from".
  For most corpora no example-level assay id exists (a DUD-E decoy was never assayed),
  so this is the only available meaning, but it is weaker than the "Same assay
  identifier" of proposal Table 2 and must not be presented as that.
- **`ligand_similar` is thresholded at Tanimoto ≥ 0.80.** Recording every pair is
  O(N²) over 2 M ligands. The value is in `props`, so a downstream step can raise the
  threshold, never lower it.
- **One obsolete UniProt accession** (`A0A8C0LZB8`) has no sequence and no cluster. No
  Example reaches it, so protein-axis coverage is still 100 %.
