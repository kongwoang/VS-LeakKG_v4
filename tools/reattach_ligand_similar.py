"""Reattach persisted ligand-similarity edges to a freshly rebuilt kg_edges.parquet.

`build_kg` rewrites kg_edges.parquet from scratch. The ~2 hours of all-pairs Tanimoto
that `ligand_similarity` appended to it are destroyed by any rebuild — including a
rebuild for a reason with nothing to do with ligands (a relabelled split, a fixed
provenance join, a new Example prop). That cost was invisible: nothing in the repo
warned about it, so the only safe-looking move was to re-run the two hours.

It is almost always unnecessary. A `ligand_similar` edge references nothing but two
Ligand node ids, and a Ligand node id is md5(canonical SMILES) — so the edges survive
any rebuild that neither adds nor removes a ligand. This tool checks exactly that,
by hash, and refuses to guess:

    python tools/reattach_ligand_similar.py \
        --kg-nodes data/processed/kg_nodes.parquet \
        --kg-edges data/processed/kg_edges.parquet

Exit 0 = reattached. Exit 2 = the ligand set moved; re-run ligand_similarity for real.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from vsleakkg.ligand_similarity import append_to_kg, ligand_set_fingerprint

log = logging.getLogger("vsleakkg.reattach")

SIMILARITY_TYPES = ("ligand_similar", "ligand_fingerprint_exact")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-nodes", type=Path, default=Path("data/processed/kg_nodes.parquet"))
    ap.add_argument("--kg-edges", type=Path, default=Path("data/processed/kg_edges.parquet"))
    ap.add_argument("--cache", type=Path, default=None,
                    help="default: <kg-edges dir>/ligand_similar_edges.parquet")
    a = ap.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    cache = a.cache or a.kg_edges.parent / "ligand_similar_edges.parquet"
    if not cache.exists():
        log.error("no cache at %s — run `python -m vsleakkg.ligand_similarity` first", cache)
        return 2

    cached = pl.read_parquet(cache)
    want = ligand_set_fingerprint(a.kg_nodes)
    got = cached["_ligand_set_sha256"][0] if "_ligand_set_sha256" in cached.columns else None

    if got != want:
        log.error("REFUSING to reattach: the ligand set has changed.")
        log.error("  edges were computed against ligand set %s", (got or "<untagged>")[:16])
        log.error("  the current kg_nodes.parquet has          %s", want[:16])
        log.error("These edges would reference ligands that no longer exist, or miss "
                  "ligands that now do. Re-run ligand_similarity.")
        return 2

    new = cached.drop("_ligand_set_sha256")
    n_by_type = dict(new.group_by("edge_type").len().iter_rows())
    total = append_to_kg(a.kg_edges, new)
    log.info("reattached %d edges (%s); kg_edges.parquet now has %d",
             new.height, n_by_type, total)

    check = pl.scan_parquet(a.kg_edges).filter(
        pl.col("edge_type").is_in(list(SIMILARITY_TYPES))
    ).select(pl.len()).collect().item()
    if check != new.height:
        log.error("post-write check failed: %d similarity edges in the file, expected %d",
                  check, new.height)
        return 1
    log.info("verified: %d similarity edges present", check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
