"""Load DUD-E `actives_final.ism` / `decoys_final.ism` for each target directory."""
from __future__ import annotations

from pathlib import Path
from typing import List

import polars as pl

from . import io as vsio


def discover_targets(raw_dir: Path) -> List[str]:
    targets = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue
        if (entry / "actives_final.ism").exists() or (entry / "decoys_final.ism").exists():
            targets.append(entry.name)
    return targets


def load_target(raw_dir: Path, target: str) -> pl.DataFrame:
    tdir = raw_dir / target
    frames = []
    for fname, label, label_type in (
        ("actives_final.ism", 1, "active"),
        ("decoys_final.ism", 0, "decoy"),
    ):
        p = tdir / fname
        if not p.exists() or p.stat().st_size == 0:
            continue
        frames.append(vsio.read_smi_dataframe(
            p, source="DUD-E", label=label, label_type=label_type,
            target=target, split="unknown",
        ))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def load_all(raw_dir: Path) -> pl.DataFrame:
    targets = discover_targets(raw_dir)
    if not targets:
        raise RuntimeError(f"No DUD-E targets discovered under {raw_dir}")
    parts = [load_target(raw_dir, t) for t in targets]
    parts = [p for p in parts if p.height > 0]
    return pl.concat(parts, how="vertical_relaxed")
