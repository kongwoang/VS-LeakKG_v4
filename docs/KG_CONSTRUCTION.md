# How the knowledge graph is built

Every decision baked into the graph, and why. Read `CONTEXT.md` first for what the
graph is *for*.

---

## 1. The design rule

**The KG records facts. It does not record policy.**

A fact is something the data states: *DUD-E's `gcr` target is UniProt P04150.* *These
two ligands have ECFP4 Tanimoto 0.91.* *This assay contains 330,122 compounds.*

A policy is a judgement about how to use a fact: *an assay with 330,122 compounds is
too promiscuous to count as leakage.* *Scaffolds under 7 heavy atoms are not
evidence.* *Cap each ligand at 5 assays so the join fits in memory.*

Policies belong to the step that scores contamination or builds splits, because a
policy can be changed there — a fact that was never recorded cannot be recovered
anywhere. This rule is why the build no longer caps, filters or flags anything.

---

## 2. Sources

Five benchmark corpora and two reference databases.

| corpus | actives | inactives | ratio | unique ligands | targets |
|---|---:|---:|---:|---:|---:|
| LIT-PCBA | 7,955 | 2,644,022 | 332 : 1 | 382,742 | 15 |
| DUD-E | 22,805 | 1,411,210 | 62 : 1 | 1,200,431 | 102 |
| BigBind | 489,733 | 93,224 | 0.19 : 1 (*it is a training set*) | 399,090 | 1,173 |
| BayesBind | 10,876 | 250,000 | 23 : 1 | 21,037 | 50 |
| DEKOIS | 3,239 | 92,429 | 28 : 1 | 87,954 | 81 |

Reference: **ChEMBL 35** (SQLite) and **BindingDB**.

PDBBind was dropped in the v3 redesign (it had been the protein anchor); BigBind and
BayesBind were added.

**Two facts worth knowing before trusting any result:** BigBind maps **99.99 %** onto
ChEMBL and BayesBind **99.98 %**. They are effectively subsets of ChEMBL, so any
ChEMBL-pretrained model has already seen them.

---

## 3. Pipeline

```
data/raw/{ChEMBL,BindingDB,DEKOIS,DUD-E,LIT-PCBA,BigBind,BayesBind}/
        │
        │  load_*.py — one loader per corpus, shared build_graph.make_nodes_edges
        ▼
data/processed/<corpus>_{examples,nodes,edges}.parquet
        │
        │  vsleakkg.build_kg — merge corpora, layer ChEMBL/BindingDB cross-refs
        ▼
data/processed/kg_{nodes,edges}.parquet                       ← raw KG
        │
        ├─ vsleakkg.ligand_similarity      → ligand_similar / ligand_fingerprint_exact
        ├─ tools/resolve_targets.py        → target_uniprot_map.parquet
        ├─ tools/build_protein_axis.py     → protein_id_map + protein_clusters_{30,50,90}
        │
        │  vsleakkg.kg.consolidate — canonical, axis-aligned graph
        ▼
outputs/kg/canonical_{nodes,edges}.parquet                    ← THE KG
        │
        │  tools/audit_kg.py   +   tools/audit_semantics.py
        ▼
verified
```

### Rebuild

Only needed if the graph changes; the shipped `outputs/kg/` is current.

```bash
export PYTHONPATH=src
PY=/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/bin/python

$PY -m vsleakkg.build_kg                     # raw KG, ~25 min on 32 cores
$PY -m vsleakkg.ligand_similarity \
      --kg-nodes data/processed/kg_nodes.parquet \
      --kg-edges data/processed/kg_edges.parquet \
      --threshold 0.70 --workers 32          # ~30–45 min
$PY tools/resolve_targets.py                 # 198 targets → UniProt (needs network)
$PY tools/build_protein_axis.py              # sequences + MMseqs2 @ 30/50/90 %
$PY -m vsleakkg.kg.consolidate --output-dir outputs/kg --corpus all   # ~5 min
$PY tools/audit_kg.py && $PY tools/audit_semantics.py
```

The build is **deterministic**: two runs over the same input produce content-identical
parquet. It also **refuses to write** a graph that fails validation — see §6.

---

## 4. The protein axis (the hard part)

The inherited graph was **blind to 81.7 % of examples** on the protein axis, and the
reason is worth understanding because it is invisible unless you look for it.

DUD-E, DEKOIS and LIT-PCBA name their targets by **gene symbol**, so their loaders
emit `tgt:<Corpus>:<gene>` nodes. MMseqs2 clusters only UniProt-anchored
`protein:<accession>` nodes, because only those have sequences. So the primary target
of essentially every example in the three headline benchmarks sat **outside the
clustering entirely** — and the one corpus with full protein coverage was BigBind,
the *training* set.

`tools/resolve_targets.py` maps all 198 benchmark targets to UniProt using **two
independent routes that must agree**:

- **DUD-E** — the target code *is* the UniProt entry name (`gcr` → `GCR_HUMAN`),
  verified independently by aligning the sequence of each downloaded `receptor.pdb`
  against UniProt. 97/102 match `_HUMAN` directly; the rest are non-human (AmpC, HIV,
  influenza) or renamed.
- **DEKOIS** — target names follow no convention, so identity comes from the
  **amino-acid sequence** in `*_protein.pdb`, aligned with MMseqs2. This is immune to
  the sibling-protein trap: COX-1 and COX-2 are only ~60 % identical, so a 99 % hit is
  unambiguous.
- **LIT-PCBA** — PDB codes in the structure filenames (reading **every** polymer
  entity, not just the first) intersected with the gene symbol.

197/198 resolved. `tools/build_protein_axis.py` then fetches every sequence (ChEMBL's
11,385 + UniProt REST for the rest) and re-runs MMseqs2 at 30/50/90 %, so **all 5,466
proteins are clustered**.

### Three routes that were tried and rejected

Recorded because each produced plausible, **wrong** answers — the kind that pass every
invariant check and quietly corrupt a result.

- **Voting on ChEMBL activities** ("which target do this DUD-E target's actives
  bind?") is wrong 30–40 % of the time. Selective ligands are assayed against *both*
  siblings, so the vote splits: `pgh1` came out as COX-2, `try1` as thrombin, `esr2`
  as ESR1.
- **Reading only polymer entity 1** of a PDB structure made LIT-PCBA's `MTORC1`
  resolve to FKBP1A — the co-crystallised partner, not the target.
- **Full-text UniProt search** for the stale entry name `GLCM_HUMAN` returns
  `GLCM1_HUMAN` (Q8IVK1, "cell adhesion molecule 1") — a different protein entirely.
  The correct target is P04062 (GBA1_HUMAN, "lysosomal acid glucosylceramidase"),
  confirmed three ways: its length is exactly 536 aa, matching the BayesBind construct
  `GLCM_HUMAN_40_536_0`; DUD-E's `glcm` aligns to it; and DEKOIS `gba` and LIT-PCBA
  `GBA` resolve to it independently.

### HIV is split by domain

DUD-E's `hivpr`, `hivrt` and `hivint` all sit on one UniProt entry — they are three
functional domains of the Gag-Pol polyprotein. Collapsing them into one node would
**overstate** leakage: a model that has learned protease has not thereby learned
reverse transcriptase. They are split into `protein:HIV1:PR` / `:RT` / `:IN`, each
carrying the chain sequence UniProt annotates for it (99 / 560 / 288 aa). DEKOIS
`hiv1pr` lands on the same protease node, exposing that cross-corpus overlap. The
audit checks they stay in separate clusters at all three resolutions.

### Clusters are forced to nest

MMseqs2 is run independently per resolution, which does not guarantee that a 90 %
cluster sits inside one 30 % cluster — and 7 of them did not. Two proteins ≥ 90 %
identical are certainly ≥ 30 % identical, so the violation is an artefact, and it
would break the monotonicity a three-level strictness sweep assumes. A union-find pass
forces **90 % ⊆ 50 % ⊆ 30 %**.

---

## 5. What is recorded, and what deliberately is not

### `example_has_protein` means the target — and only the target

It used to carry two different claims at once: *"this example's TARGET is P"*
(5,025,493 edges, from the corpus loader) and *"this example's LIGAND was once
measured against P"* (2,890,812 edges, from the BindingDB wire). Those are not the
same statement, and merging them was not a rounding error: 550,031 examples ended up
with several proteins each, every one a bridge, and the protein axis collapsed into
**one leakage group covering 100 % of examples** — a protein-clean split was
arithmetically impossible and nothing said so.

The BindingDB relation is kept, retyped **`ligand_measured_protein`** (Ligand →
Protein) and left **out of every leakage axis**. It is real evidence of *pretraining*
contamination — a ChEMBL/BindingDB-trained model has seen that (ligand, protein) pair
— which is a different question from benchmark-split leakage and should be asked
separately, not smuggled into the protein axis.

### Scaffolds are keyed on chemistry, not on spelling

A Bemis-Murcko scaffold is a **topological** object: two stereoisomers share it, and a
model that memorises one recognises the other. Scaffold nodes are keyed on the
RDKit-canonical **stereo-free** SMILES, which merged 76,165 duplicate nodes.

(String-stripping `/`, `\` and `@` is *not* sufficient — two molecules with the same
flat framework can have canonical SMILES whose atom **order** differs, so the stripped
strings differ and the nodes never meet. That method missed 9,049 of them.)

### No caps, no filters, no flags

Four policies were removed from the build:

- **The assay cap.** Each benchmark ligand used to be capped at 5 ChEMBL assays, "to
  bound memory". It silently truncated **71 %** of examples, and it truncated with a
  systematic bias: the kept assays were the five with the smallest ChEMBL id, i.e. the
  **oldest**. Two ligands sharing a recent assay were simply never linked. Uncapped,
  `example_from_assay` goes from 10.3 M to **51.6 M** edges. Memory is cheap; a fact
  that was never recorded is not recoverable.
- **`is_hub`.** A boolean set by `degree > 1000` — an arbitrary threshold frozen into
  the data. "How informative is sharing this node" is a **continuous** quantity (an
  assay with 4 compounds is strong evidence; an HTS screen with 330,122 is none), and
  a boolean throws that away. The graph now records **`degree`** and lets the
  downstream step choose its own cut-off — or discount weights continuously instead of
  excluding at all.
- **The trivial-scaffold filter.** Scaffolds with ≤ 6 heavy atoms were deleted. But
  benzene **is** a scaffold; calling it weak evidence is an interpretation. Nodes are
  kept, with `props.n_heavy_atoms` recorded.
- **The decoy-protocol grouping.** "DUD-E and DEKOIS both use property-matched decoys,
  therefore they share a protocol" is an *inference*, not something the data states.
  It is now two tiers, so both facts are recorded separately and the downstream step
  chooses whether to traverse the second:

  ```
  Example(inactive) --source_decoy_protocol--> proto:DUD-E
  proto:DUD-E       --decoy_protocol_in_class--> protoclass:property_matched
  proto:DEKOIS      --decoy_protocol_in_class--> protoclass:property_matched
  ```

  Only **inactives** are linked — an active compound is not a product of a
  decoy-generation procedure. LIT-PCBA and BigBind inactives are *experimentally
  measured*, not generated, so they get **no protocol edge at all**: linking them
  would fabricate a leakage path where none exists.

### The one approximation that cannot be removed

`ligand_similar` records only pairs with ECFP4 Tanimoto **≥ 0.80**. A truly complete
graph would record every pair, but that is O(N²) over 2 M ligands. This is a forced
approximation, unlike the assay cap which was merely convenient. The Tanimoto value is
stored in `props` so a downstream step can raise the threshold, never lower it.

Similarly, protein clusters exist at exactly 30 / 50 / 90 % identity — a choice of
resolution, but all three are recorded, so the downstream step has what it needs.

---

## 6. Two things that protect the graph

### It refuses to write a corrupt graph

polars on this box **intermittently corrupts string columns** under load. Two
consecutive runs of identical code over identical input produced 38,549,309 edges with
`edge_type` values like `'\x00\x00\x00mple_has_ligand'`, and 38,547,837 edges clean.
It is the same null-byte bug the v3 history records against `task_build_kg` — which
grew write-time invariants in response. `consolidate` never did, so a corrupt graph
could be written and audited later, or not at all.

`fixes.validate_canonical()` now runs immediately before the parquet write and raises
on NUL bytes, unknown node/edge types, duplicate ids or dangling edges. **A bad run
dies instead of shipping a graph whose leakage numbers would be quietly wrong.**

### It is deterministic

`unique(keep="first")` is **not** deterministic — polars is multithreaded, so which of
two duplicate rows arrives "first" varies between runs, and with it the `label` and
`props` that survive. Two builds produced the same node and edge *sets* with different
labels, which means a rebuild could never be verified against the shipped graph. All
dedup now sorts on every column first (`fixes.stable_unique`), and the frames are
sorted before writing.
