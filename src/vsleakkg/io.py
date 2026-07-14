"""Lightweight readers for .ism / .smi and a thin parquet writer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import polars as pl


def iter_ism(path: Path) -> Iterator[Tuple[str, Optional[str], Optional[str]]]:
    """Yield (smiles, id1, id2) per non-empty line. Lines are whitespace-split.
    Works for DUD-E `.ism` (smiles  zinc_id  chembl_id) and the plain `.smi`
    variants used by LIT-PCBA (smiles  id)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            id1 = parts[1] if len(parts) >= 2 else None
            id2 = parts[2] if len(parts) >= 3 else None
            yield smi, id1, id2


def read_smi_dataframe(path: Path, *, source: str, label: Optional[int],
                       label_type: Optional[str], target: Optional[str],
                       split: Optional[str]) -> pl.DataFrame:
    rows = list(iter_ism(path))
    if not rows:
        return pl.DataFrame(schema={
            "smiles_input": pl.Utf8, "ext_id_1": pl.Utf8, "ext_id_2": pl.Utf8,
            "source": pl.Utf8, "target": pl.Utf8, "label": pl.Int8,
            "label_type": pl.Utf8, "split": pl.Utf8,
        })
    return pl.DataFrame({
        "smiles_input": [r[0] for r in rows],
        "ext_id_1":     [r[1] for r in rows],
        "ext_id_2":     [r[2] for r in rows],
        "source":       [source] * len(rows),
        "target":       [target] * len(rows),
        "label":        [label]  * len(rows),
        "label_type":   [label_type] * len(rows),
        "split":        [split]  * len(rows),
    }, schema={
        "smiles_input": pl.Utf8, "ext_id_1": pl.Utf8, "ext_id_2": pl.Utf8,
        "source": pl.Utf8, "target": pl.Utf8, "label": pl.Int8,
        "label_type": pl.Utf8, "split": pl.Utf8,
    })


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
