"""Regression tests for vsleakkg.chem.

Pin the canonical SMILES / InChIKey / scaffold / parent contracts that the
KG ligand identifiers depend on. If RDKit (or our wrapper) starts producing
different canonical output, these tests catch it before a build silently
splits a single Ligand into two nodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vsleakkg import chem as vc


# ---------------------------------------------------------------------------
# canonical SMILES + InChIKey + scaffold determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("smi,canon,ik,scaf", [
    # aspirin: free base. Murcko scaffold is bare benzene.
    ("CC(=O)OC1=CC=CC=C1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O",
     "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "c1ccccc1"),
    # benzene — its own scaffold.
    ("c1ccccc1", "c1ccccc1", "UHOVQNZJYSORNB-UHFFFAOYSA-N", "c1ccccc1"),
    # ethanol — acyclic, scaffold is the empty SMILES.
    ("CCO", "CCO", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N", ""),
])
def test_featurize_pinned(smi: str, canon: str, ik: str, scaf: str):
    """If any of these change, the KG's `lig:md5(canonical_smiles)` ids drift
    and we silently split a single chemical into two Ligand nodes."""
    feats = vc.featurize(smi)
    assert feats.parse_ok, smi
    assert feats.smiles_canonical == canon, (
        f"canonical drift: got {feats.smiles_canonical!r}, expected {canon!r}")
    assert feats.inchikey == ik
    assert feats.scaffold_smiles == scaf


def test_invalid_smiles_parse_ok_false():
    feats = vc.featurize("not-a-smiles")
    assert feats.parse_ok is False
    assert feats.smiles_canonical is None
    assert feats.inchikey is None


# ---------------------------------------------------------------------------
# salt-stripped parent InChIKey
# ---------------------------------------------------------------------------


def test_parent_inchikey_aspirin_hcl_equals_free_base():
    """Aspirin.HCl and aspirin must collapse to the same parent."""
    pk_hcl = vc.parent_inchikey("CC(=O)OC1=CC=CC=C1C(=O)O.Cl")
    pk_free = vc.parent_inchikey("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert pk_hcl == pk_free
    assert pk_hcl == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_parent_inchikey_no_salt_round_trip():
    """For a salt-free molecule the parent InChIKey equals the regular InChIKey."""
    smi = "CN1CCC23C4OC5=C(O)C=CC(CC1C2C=CC4O)=C35"   # morphine
    assert vc.parent_inchikey(smi) == vc.inchikey(smi)


def test_parent_inchikey_invalid_returns_none():
    assert vc.parent_inchikey("not-a-smiles") is None
    assert vc.parent_inchikey("") is None
    assert vc.parent_inchikey(None) is None


# ---------------------------------------------------------------------------
# Parallel batch order preservation + length contract
# ---------------------------------------------------------------------------


def test_featurize_batch_parallel_order_preserved():
    base = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCN", "c1ccccc1", "not-a-smiles", "CCO"]
    smis = base * 200          # 1000 entries — triggers the parallel path
    out = vc.featurize_batch_parallel(smis, n_workers=4)
    assert len(out) == len(smis)
    for i, expected in enumerate(smis):
        assert out[i].smiles_input == expected, f"order broken at index {i}"


def test_featurize_batch_parallel_sequential_fallback():
    """Below chunksize threshold falls back to the sequential path."""
    smis = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCN", "c1ccccc1"]
    out = vc.featurize_batch_parallel(smis, n_workers=4, chunksize=10000)
    assert [r.smiles_input for r in out] == smis


def test_parent_inchikey_batch_parallel_order_preserved():
    smis = ["CC(=O)OC1=CC=CC=C1C(=O)O.Cl",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "not-a-smiles",
            "CCO"] * 1000
    out = vc.parent_inchikey_batch_parallel(smis, n_workers=4)
    assert len(out) == len(smis)
    # Per-index value contract:
    assert out[0] == out[1]
    assert out[2] is None
    # Length stays exactly aligned across full batch:
    assert out == out[: len(out)]
