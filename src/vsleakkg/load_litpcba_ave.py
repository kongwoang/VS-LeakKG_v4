"""Load LIT-PCBA AVE_unbiased train/validation splits.

Layout:
    data/raw/LIT-PCBA/splits/AVE_unbiased/<TARGET>/
        active_T.smi      label=1, split=train
        active_V.smi      label=1, split=validation
        inactive_T.smi    label=0, split=train
        inactive_V.smi    label=0, split=validation

Each .smi line is `SMILES  PUBCHEM_CID`. Splits are AVE-debiased (Tran-Nguyen
et al. 2020).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl

from . import io as vsio


LITPCBA_TARGETS = (
    "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",
    "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
)

_SPLIT_FILES = (
    ("active_T.smi",   1, "active",   "train"),
    ("active_V.smi",   1, "active",   "validation"),
    ("inactive_T.smi", 0, "inactive", "train"),
    ("inactive_V.smi", 0, "inactive", "validation"),
)


def discover_targets(root: Path) -> List[str]:
    return [t for t in LITPCBA_TARGETS if (root / t).exists()]


def load_target(root: Path, target: str) -> pl.DataFrame:
    tdir = root / target
    frames = []
    for fname, label, label_type, split in _SPLIT_FILES:
        p = tdir / fname
        if not p.exists() or p.stat().st_size == 0:
            continue
        df = vsio.read_smi_dataframe(
            p, source="LIT-PCBA", label=label, label_type=label_type,
            target=target, split=split,
        )
        if df.height == 0:
            continue
        df = df.with_columns(pl.lit(fname).alias("source_file"))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def load_all(root: Path) -> pl.DataFrame:
    """Load every target under `root` (the AVE_unbiased extraction)."""
    targets = discover_targets(root)
    if not targets:
        raise RuntimeError(f"No LIT-PCBA AVE targets discovered under {root}")
    parts = [load_target(root, t) for t in targets]
    parts = [p for p in parts if p.height > 0]
    return pl.concat(parts, how="vertical_relaxed")
