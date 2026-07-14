"""Exact pairwise ligand similarity edges via bit-bound pruning.

Computes Morgan ECFP4 (radius=2, 2048 bits) for every Ligand in the KG and
emits `ligand_similar` edges for every pair with Tanimoto >= threshold.

Brute-force pairwise is O(N^2) = 4e12 for N=2M ligands → infeasible. We use
the **Swamidass & Baldi 2007** bit-bound pruning:

    Tanimoto(A, B) <= min(|A|, |B|) / max(|A|, |B|)

where |A| = popcount of fingerprint A. By sorting ligands by popcount and
walking the sorted order, for each query Q we only need to compare against
target T whose popcount |T| satisfies |Q| * threshold <= |T| <= |Q| / threshold.
For Tanimoto >= 0.80 the eligible popcount window is roughly +/-25% of |Q|,
collapsing the work by 1-2 orders of magnitude.

THRESHOLD — read before changing it. The KG declares `ligand_similar` to cover
[0.80, 0.9995). The shipped graph did not: the global pass had actually been run
at **0.85**, and the only edges below it were 811 legacy `ligand_similar_to_ligand`
rows emitted by the LIT-PCBA loader alone. So the band [0.80, 0.85) held 427 edges
where the density of the distribution says it should hold several hundred thousand
(bin 0.85 alone has 116,990) — and every one of those 427 was a LIT-PCBA pair. The
primary leakage axis was blind in a whole band, and blind asymmetrically, in the
band where cross-corpus ligand overlap most needed to be seen. The default is now
0.80 and it matches the declared contract. Lower it if you want more; never raise
it without changing the contract in `kg/schema.py` and the docs together.

Within the eligible window we use `rdkit.DataStructs.BulkTanimotoSimilarity`
which is C-vectorized — ~50M comparisons/second per core.

Parallelism: query ligands are split into chunks; each worker handles its
chunk against the full sorted target array (read-only memory shared via
multiprocessing fork). Workers stream rows back; the main process appends to
the edge buffer in input-order-agnostic fashion (we are emitting symmetric
edges, so order doesn't matter — but we always emit src < dst lexicographically
for downstream dedup).

CLI:
    PYTHONPATH=src python -m vsleakkg.ligand_similarity \\
        --kg-nodes data/processed/kg_nodes.parquet \\
        --kg-edges data/processed/kg_edges.parquet \\
        --threshold 0.80 \\
        --workers 32
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

log = logging.getLogger("vsleakkg.ligand_similarity")


def _fp_from_smiles(smi: str) -> "DataStructs.ExplicitBitVect | None":
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _fp_pack_worker(args: tuple[int, list[str]]) -> tuple[int, list[bytes | None]]:
    """Return (start_index, list of fingerprint binary blobs).

    The blobs are RDKit BitVect binary text — fast to round-trip via
    `DataStructs.CreateFromBinaryText`. Returning blobs (not BitVect objects)
    is cheaper to pickle across the multiprocessing boundary.
    """
    start, smis = args
    out: list[bytes | None] = []
    for s in smis:
        fp = _fp_from_smiles(s)
        out.append(DataStructs.BitVectToBinaryText(fp) if fp is not None else None)
    return start, out


def _restore_fp(blob: bytes):
    return DataStructs.CreateFromBinaryText(blob)


def _similar_chunk(args: tuple) -> list[tuple[int, int, float]]:
    """Worker: compare a chunk of query indices against the global target arrays.

    Each query Q at sorted index q is compared against targets at indices in
    `eligible_window(popcount[q])`. We pass the precomputed sorted arrays
    (popcount, fp blobs, lig_ids) as workers' globals through fork().
    """
    q_indices, threshold = args
    # Pull module globals set in `_init_worker` — fps is now a list of
    # already-restored BitVect objects so the inner loop is comparison-only.
    pc = _WORKER_POPCOUNTS
    fps = _WORKER_FPS
    pairs: list[tuple[int, int, float]] = []
    for q in q_indices:
        pq = pc[q]
        lo_pc = max(1, int(threshold * pq))
        hi_pc = int(pq / threshold) if threshold > 0 else len(pc)
        lo = np.searchsorted(pc, lo_pc, side="left")
        hi = np.searchsorted(pc, hi_pc, side="right")
        if hi <= q:
            continue
        start = max(lo, q + 1)
        if start >= hi:
            continue
        # No restoration inside the loop — direct slice into the cached fps.
        sims = DataStructs.BulkTanimotoSimilarity(fps[q], fps[start:hi])
        for off, s in enumerate(sims):
            if s >= threshold:
                pairs.append((q, start + off, float(s)))
    return pairs


_WORKER_POPCOUNTS = None
_WORKER_FPS = None        # list[BitVect] — pre-restored, NOT raw blobs
_WORKER_LIG_IDS = None


def _init_worker(pc, fp_blobs, lids):
    """One-time-per-worker setup: restore every fingerprint blob into a
    BitVect object. This trades memory (~600 MB per worker for 2 M ligands)
    for ~50x speedup in the inner loop, which would otherwise call
    `_restore_fp` once per comparison."""
    global _WORKER_POPCOUNTS, _WORKER_FPS, _WORKER_LIG_IDS
    _WORKER_POPCOUNTS = pc
    _WORKER_FPS = [DataStructs.CreateFromBinaryText(b) for b in fp_blobs]
    _WORKER_LIG_IDS = lids


def compute_ligand_similar_edges(
    kg_nodes_path: Path,
    threshold: float = 0.80,
    n_workers: int | None = None,
    chunk_size: int = 200,
) -> pl.DataFrame:
    """Return a polars DataFrame of (src, dst, edge_type, props) rows.

    src and dst are KG Ligand node_ids; src < dst lexicographically so the
    downstream `unique()` on the (src, dst, edge_type) tuple collapses
    duplicate emissions.
    """
    nodes = pl.read_parquet(kg_nodes_path)
    ligs = (nodes.filter(pl.col("node_type") == "Ligand")
                .select(["node_id", "label"])
                .filter(pl.col("label").is_not_null() & (pl.col("label") != ""))
                )
    log.info("compute_ligand_similar_edges: %d Ligand nodes", ligs.height)
    if ligs.is_empty():
        return pl.DataFrame(schema=["src", "dst", "edge_type", "props"])

    smis = ligs["label"].to_list()
    lids = ligs["node_id"].to_list()
    n = len(smis)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    # 1) Parallel fingerprint computation.
    log.info("computing ECFP4 fingerprints (parallel, %d workers) ...", n_workers)
    t0 = time.perf_counter()
    chunks = []
    cs = max(1, n // (n_workers * 4))   # smaller chunks for fingerprint = better balance
    for i in range(0, n, cs):
        chunks.append((i, smis[i:i + cs]))
    fp_blobs: list[bytes | None] = [None] * n
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for start, blobs in pool.imap_unordered(_fp_pack_worker, chunks, chunksize=1):
            for j, b in enumerate(blobs):
                fp_blobs[start + j] = b
    fp_time = time.perf_counter() - t0
    n_parsed = sum(1 for b in fp_blobs if b is not None)
    log.info("fingerprints: %d/%d parsed in %.1fs", n_parsed, n, fp_time)

    # 2) Drop unparsed entries and compute popcounts (number of set bits).
    keep = [i for i, b in enumerate(fp_blobs) if b is not None]
    fp_blobs = [fp_blobs[i] for i in keep]
    lids = [lids[i] for i in keep]
    popcounts = np.fromiter(
        (_restore_fp(b).GetNumOnBits() for b in fp_blobs),
        dtype=np.int32, count=len(fp_blobs))

    # 3) Sort everything by popcount ascending.
    order = np.argsort(popcounts, kind="stable")
    popcounts = popcounts[order]
    fp_blobs = [fp_blobs[i] for i in order]
    lids = [lids[i] for i in order]
    log.info("sorted by popcount; range = %d..%d, median = %d",
             int(popcounts.min()), int(popcounts.max()), int(np.median(popcounts)))

    # 4) Parallel pairwise comparison via bit-bound pruning.
    log.info("pairwise Tanimoto via bit-bound (threshold %.2f) ...", threshold)
    t1 = time.perf_counter()
    q_chunks: list[tuple[list[int], float]] = []
    for i in range(0, len(popcounts), chunk_size):
        q_chunks.append((list(range(i, min(i + chunk_size, len(popcounts)))), threshold))
    log.info("pairwise: %d chunks of %d queries each", len(q_chunks), chunk_size)
    pairs: list[tuple[int, int, float]] = []
    n_chunks_done = 0
    log_every = max(1, len(q_chunks) // 50)
    with mp.get_context("fork").Pool(
        n_workers, initializer=_init_worker,
        initargs=(popcounts, fp_blobs, lids)) as pool:
        for chunk_pairs in pool.imap_unordered(_similar_chunk, q_chunks, chunksize=1):
            pairs.extend(chunk_pairs)
            n_chunks_done += 1
            if n_chunks_done % log_every == 0 or n_chunks_done == len(q_chunks):
                elapsed = time.perf_counter() - t1
                pct = n_chunks_done / len(q_chunks)
                eta = elapsed * (1 - pct) / pct if pct > 0 else float("inf")
                log.info("  chunks %d/%d (%.1f%%), pairs so far: %d, "
                         "elapsed %.0fs, ETA %.0fs",
                         n_chunks_done, len(q_chunks), 100*pct, len(pairs),
                         elapsed, eta)
    sim_time = time.perf_counter() - t1
    log.info("found %d similar pairs in %.1fs", len(pairs), sim_time)

    # 5) Convert to edge DataFrame: ensure src < dst lexicographically.
    # Pairs with Tanimoto == 1.0 are tagged `ligand_fingerprint_exact` rather
    # than `ligand_similar` — they're the same molecule modulo stereo /
    # tautomer detail that ECFP4 (radius=2) doesn't encode, so the audit
    # should treat them as near-identity (weight 0.95) instead of weak
    # similarity (weight 0.65). T < 1.0 stays as `ligand_similar`.
    rows = []
    import json
    n_fp_exact = 0
    n_similar = 0
    for q, t, s in pairs:
        a, b = lids[q], lids[t]
        if a > b:
            a, b = b, a
        et = "ligand_fingerprint_exact" if s >= 0.9995 else "ligand_similar"
        if et == "ligand_fingerprint_exact":
            n_fp_exact += 1
        else:
            n_similar += 1
        rows.append((a, b, et,
                     json.dumps({"tanimoto": round(s, 4),
                                  "fp_type": "ECFP4_2048bit",
                                  "method": "bit_bound_exact"})))
    edges = pl.DataFrame(rows, schema=["src", "dst", "edge_type", "props"],
                         orient="row").unique(subset=["src", "dst", "edge_type"])
    log.info("emitted %d ligand_fingerprint_exact + %d ligand_similar edges (%d total after dedup)",
             n_fp_exact, n_similar, edges.height)
    return edges


def append_to_kg(edges_path: Path, new_edges: pl.DataFrame) -> int:
    """Append the new edges to the existing kg_edges parquet (with dedup).

    Drops any prior `ligand_similar` and `ligand_fingerprint_exact` edges
    first so a re-run with different thresholds / split logic fully
    overwrites the previous output instead of accumulating.
    """
    existing = pl.read_parquet(edges_path)
    # `ligand_similar_to_ligand` is the per-corpus loaders' name for the same
    # relation. It was NOT in this drop set, so 811 legacy LIT-PCBA rows survived
    # every re-run and mixed a second, lower threshold into an edge type that is
    # supposed to have exactly one.
    drop = {"ligand_similar", "ligand_fingerprint_exact", "ligand_similar_to_ligand"}
    existing = existing.filter(~pl.col("edge_type").is_in(list(drop)))
    merged = pl.concat([existing, new_edges], how="vertical_relaxed").unique()
    merged.write_parquet(edges_path)
    return merged.height


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-nodes", type=Path, required=True)
    p.add_argument("--kg-edges", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(message)s")
    edges = compute_ligand_similar_edges(
        args.kg_nodes, threshold=args.threshold,
        n_workers=args.workers, chunk_size=args.chunk_size)
    total = append_to_kg(args.kg_edges, edges)
    log.info("kg_edges.parquet now has %d total edges", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
