"""Load LIT-PCBA `full_data` archive into a normalized Polars frame.

The official archive ships per-target folders, each containing:
- actives.smi      (SMILES, PubChem CID)        — label 1
- inactives.smi    (SMILES, PubChem CID)        — label 0
- <pdb>_ligand.mol2 / <pdb>_protein.mol2 pairs   — query ligands + receptors
- no train/val/test split files in this archive.

Split labels are therefore left as `unknown` and `split_source` records why.
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Optional

import polars as pl

from . import io as vsio


LITPCBA_TARGETS = (
    "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A",
    "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
)


def ensure_extracted(raw_dir: Path) -> Path:
    """Extract `full_data.tgz` into `raw_dir/extracted/` if not already done.
    Returns the path to the extracted root."""
    archive = raw_dir / "full_data.tgz"
    out = raw_dir / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    sentinel = out / ".extracted_ok"
    if sentinel.exists():
        return out
    if not archive.exists():
        raise FileNotFoundError(f"LIT-PCBA archive missing: {archive}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out)
    sentinel.write_text("")
    return out


def discover_targets(extracted: Path) -> List[str]:
    targets = []
    for entry in sorted(extracted.iterdir()):
        if entry.is_dir() and entry.name in LITPCBA_TARGETS:
            targets.append(entry.name)
    return targets


def list_query_ligand_pdbs(extracted: Path, target: str) -> List[str]:
    tdir = extracted / target
    pdbs: List[str] = []
    if not tdir.exists():
        return pdbs
    for f in tdir.glob("*_ligand.mol2"):
        pdbs.append(f.stem.split("_ligand")[0])
    return sorted(set(pdbs))


def load_target(extracted: Path, target: str) -> pl.DataFrame:
    """Load actives + inactives for one target. Returns a frame with columns
    smiles_input, ext_id_1, ext_id_2, source, target, label, label_type, split."""
    tdir = extracted / target
    frames = []
    for fname, label, label_type in (
        ("actives.smi", 1, "active"),
        ("inactives.smi", 0, "inactive"),
    ):
        p = tdir / fname
        if not p.exists():
            continue
        frames.append(vsio.read_smi_dataframe(
            p, source="LIT-PCBA", label=label, label_type=label_type,
            target=target, split="unknown",
        ))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def load_all(raw_dir: Path) -> pl.DataFrame:
    extracted = ensure_extracted(raw_dir)
    targets = discover_targets(extracted)
    if not targets:
        raise RuntimeError(f"No LIT-PCBA targets discovered under {extracted}")
    parts = [load_target(extracted, t) for t in targets]
    parts = [p for p in parts if p.height > 0]
    return pl.concat(parts, how="vertical_relaxed")
