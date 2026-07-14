"""Load DEKOIS 2.0 per-target ligand lists.

DEKOIS 2.0 distributes each target as `DEKOIS2/<target>/active_decoys.smi` —
a single file containing **both** the 40 actives and ~1100 decoys mixed.
The convention is unambiguous:

    BDB<digits>   -> active   (id sourced from BindingDB)
    ZINC<digits>  -> decoy    (id sourced from ZINC)

A handful of files may include other identifier prefixes; those rows are
labeled `unknown` and surfaced in the dataset summary rather than dropped.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import polars as pl

from . import io as vsio


def discover_targets(root: Path) -> List[str]:
    """`root` is the extraction root that contains `DEKOIS2/<target>/`."""
    base = root / "DEKOIS2"
    base = base if base.exists() else root
    targets: List[str] = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and (entry / "active_decoys.smi").exists():
            targets.append(entry.name)
    return targets


_RE_BDB  = re.compile(r"^BDB\d+", re.IGNORECASE)
_RE_ZINC = re.compile(r"^ZINC\d+", re.IGNORECASE)


def _classify(ident: str) -> tuple[int, str]:
    """Return (label, label_type) from a DEKOIS id."""
    if ident is None:
        return -1, "unknown"
    s = ident.strip()
    if _RE_BDB.match(s):
        return 1, "active"
    if _RE_ZINC.match(s):
        return 0, "decoy"
    return -1, "unknown"


def load_target(root: Path, target: str) -> pl.DataFrame:
    base = root / "DEKOIS2"
    base = base if base.exists() else root
    p = base / target / "active_decoys.smi"
    if not p.exists() or p.stat().st_size == 0:
        return pl.DataFrame()
    rows = list(vsio.iter_ism(p))
    if not rows:
        return pl.DataFrame()
    smiles = [r[0] for r in rows]
    ids    = [r[1] for r in rows]
    labels = []
    label_types = []
    for ident in ids:
        lbl, ltype = _classify(ident)
        labels.append(lbl)
        label_types.append(ltype)
    return pl.DataFrame({
        "smiles_input": smiles,
        "ext_id_1":     ids,
        "ext_id_2":     [None] * len(rows),
        "source":       ["DEKOIS"] * len(rows),
        "target":       [target] * len(rows),
        "label":        labels,
        "label_type":   label_types,
        "split":        ["unknown"] * len(rows),
        "source_file":  ["active_decoys.smi"] * len(rows),
    }, schema={
        "smiles_input": pl.Utf8, "ext_id_1": pl.Utf8, "ext_id_2": pl.Utf8,
        "source": pl.Utf8, "target": pl.Utf8, "label": pl.Int8,
        "label_type": pl.Utf8, "split": pl.Utf8, "source_file": pl.Utf8,
    })


def load_all(root: Path) -> pl.DataFrame:
    targets = discover_targets(root)
    if not targets:
        raise RuntimeError(f"No DEKOIS targets discovered under {root}")
    parts = [load_target(root, t) for t in targets]
    parts = [p for p in parts if p.height > 0]
    return pl.concat(parts, how="vertical_relaxed")
