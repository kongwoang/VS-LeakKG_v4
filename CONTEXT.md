# VS-LeakKG v4 — Context

**Read this first.** Three documents, no more:

| file | what it is |
|---|---|
| `CONTEXT.md` (this) | why the project exists, what state it is in, what to do next |
| `docs/KG_CONSTRUCTION.md` | how the knowledge graph is built, and every decision baked into it |
| `docs/KG_REPORT.md` | what the finished graph contains, how it was verified, what it says |

Source material: `docs/proposal.txt` (+ `VS_LeakKG.pdf`) — the paper proposal.

Everything lives on VUW at `/vol/dl-nguyenb5-solar/users/hoangpc/VS-LeakKG_v4`.
There is no Windows mirror and no git repo yet.

---

## 1. Why this exists

Structure-based virtual screening is increasingly posed as a **retrieval** problem:
embed protein pockets and ligands in a shared space, rank by similarity. Contrastive
models (DrugCLIP, LigUnity, S2Drug, HypSeek, ConGLUDe) report strong numbers on
DUD-E, DEKOIS and LIT-PCBA.

**Those numbers are hard to trust**, for two reasons that compound each other:

1. **The benchmarks contain shortcuts.** DUD-E-style decoys are property-matched but
   topologically dissimilar, so a *ligand-only* model can separate actives from
   decoys using chemical artifacts — no binding mechanism required.
2. **The training corpora overlap the benchmarks through indirect channels.** Models
   pretrain on ChEMBL / BindingDB / PubChem, which reach the benchmarks through
   shared ligands, similar scaffolds, homologous proteins, reused assays, common
   publications and source-specific decoy protocols. An example can be novel as a
   protein–ligand *pair* while being heavily contaminated through its scaffold, its
   protein family, or its assay provenance.

Existing work covers pieces of this. DrugOOD annotates domains; DataSAIL minimises
cross-partition similarity generically; LP-PDBBind and CleanSplit target affinity,
not screening; BayesBind holds out dissimilar targets. **None models VS contamination
as a typed, high-order relational structure.**

**The claim:** contamination is a *heterogeneous graph* problem. Examples are linked
by chemical, biological, experimental and provenance relations; leakage is a short
high-weight path in that graph. Making the graph explicit buys three things —
**audit** (which shortcut dominates, not just whether one exists), **construct**
(splits that forbid leakage *paths*, by collapsing examples joined by forbidden
relations into atomic leakage groups), and **retrain** (fit, select and test entirely
inside leakage-controlled partitions, so a score reflects generalisation rather than
overlap).

VS-LeakKG is a **benchmark-governance layer**, not another benchmark.

The contamination score (proposal §3.3), for a test example `x_t` and reference set
`A`, with edge-type leakage weights `w_r ∈ [0,1]`:

```
C(x_t, A) = max over x_i∈A, over typed paths π of length ≤ L:  S(π) = Π_{e∈π} w_r(e)
```

decomposed per axis: `C = max{C_ligand, C_scaffold, C_protein, C_pocket, C_assay,
C_source, C_time}`.

---

## 2. Where things stand

**The knowledge graph is finished, verified and deterministic.**
**No experiment has been run.** Nothing downstream exists yet — that is deliberate:
the experimental design is being rebuilt from scratch, and the graph had to be
trustworthy first.

```
outputs/kg/canonical_nodes.parquet    8,574,944 nodes
outputs/kg/canonical_edges.parquet   75,369,818 edges
outputs/kg/stats.csv
```

Two audit suites, both clean — **run them after any change to the KG**:

```bash
PYTHONPATH=src python tools/audit_kg.py         # invariants; exit code = failures
PYTHONPATH=src python tools/audit_semantics.py  # biology, chemistry, usability
```

Full numbers and verification in `docs/KG_REPORT.md`.

### The design rule the graph obeys

**The KG records facts. It does not record policy.**

Anything of the form "this evidence is too weak to count", "this node is too
promiscuous", "cap this for tractability" belongs to the *downstream* step that
scores contamination or builds splits — not to the graph. The graph's job is to state
what is true and to state it completely, so that no downstream choice is foreclosed.

This rule was applied late and it removed several things that had been silently baked
into the build: an assay cap that truncated 71 % of examples, a boolean `is_hub` flag
that froze an arbitrary degree threshold into the data, a "trivial scaffold" filter
that deleted benzene, and a decoy-protocol grouping that encoded an inference as a
fact. See `docs/KG_CONSTRUCTION.md` §5.

---

## 3. What the graph is ready for, and what it is not

Measured, not assumed (`tools/audit_semantics.py`, section C):

| axis | coverage | can it be partitioned? |
|---|---:|---|
| ligand | 100 % | yes — largest atomic block 0.1 % of the corpus |
| scaffold | 100 % | yes — 2.2 %, or 0.1 % if you exclude high-degree scaffolds |
| protein (30/50/90 %) | 100 % | yes — 7.2 % at 90 %, 8.5 % at 30 % |
| assay | 48 % | yes, but only with a degree cut-off: 44.8 % → 8.2 % |
| publication | 48 % | yes — 13.0 %, but see the caveat below |
| source / decoy | 100 % | trivially |
| **time** | **0 %** | **the axis is empty** |

Two things the experiment design must confront before writing a line of split code:

**The provenance axes are confounded with the label.** In DUD-E, 91.8 % of actives
have an assay edge and 0.9 % of decoys do; for publications it is 94.5 % vs 0.3 %.
The reason is mundane — DUD-E actives come from ChEMBL, its decoys come from ZINC and
were never assayed or published. This is simultaneously **a headline finding** (it is
exactly the source-only shortcut of proposal §3.8: you can predict the label from
provenance alone, without any protein–ligand modelling) **and a trap** (any
contamination score on the assay or publication axis will be systematically higher
for actives, so contamination-decile analyses will be confounded with the label).

**Three axes the proposal promises do not exist.** The **pocket** axis was removed in
the v3 redesign — `data/raw/DUD-E_pockets_fetched/` is preserved if it is revived.
The **time** axis is declared in the schema and carries zero edges; it needs ChEMBL
document dates. And `example_from_assay` / `example_from_publication` are
*ligand-mediated* — they mean "this example's ligand was tested in / reported in",
not "this example's label came from". For most corpora no example-level assay id
exists (a DUD-E decoy was never assayed), so this is the only available meaning, but
it is weaker than the "Same assay identifier" of proposal Table 2 and must not be
presented as that.

---

## 4. Environment

- Box: `cuda12.ecs.vuw.ac.nz`, 3 × Quadro RTX 6000, 93 GB RAM, 32 cores.
- Python: `/vol/dl-nguyenb5-solar/users/hoangpc/envs/vsleak2/bin/python` (3.12,
  polars + RDKit + scipy). MMseqs2 on `PATH` (`../bin`).
- Run modules with `PYTHONPATH=src`.
- `outputs/kg_v0/` holds the inherited v3 graph, kept for comparison.
