"""Path resolution for the canonical KG consolidator.

By default the consolidator auto-detects the repo root from the location of
`__file__`. Override via the `VSLEAKKG_ROOT` environment variable when running
from a different working directory.

    >>> from vsleakkg.kg.datapaths import data_root, processed_dir
    >>> data_root()
    PosixPath('/vol/.../VS-LeakKG_v4')
    >>> processed_dir()
    PosixPath('/vol/.../VS-LeakKG_v4/data/processed')
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "VSLEAKKG_ROOT"


def _default_root() -> Path:
    # vsleakkg/kg/datapaths.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Root of the repo (data/, outputs/, src/, ...)."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return _default_root()


def processed_dir() -> Path:
    """`data/processed/` — where build_kg writes kg_nodes/kg_edges parquets."""
    return data_root() / "data" / "processed"


def raw_dir() -> Path:
    """`data/raw/` — where the raw dataset archives live."""
    return data_root() / "data" / "raw"


def require_data_root() -> Path:
    """Like `data_root()` but raises if the directory is missing."""
    root = data_root()
    if not root.exists():
        raise FileNotFoundError(
            f"VS-LeakKG root not found at {root}. "
            f"Set the {_ENV_VAR} env var to point at your local checkout."
        )
    return root
