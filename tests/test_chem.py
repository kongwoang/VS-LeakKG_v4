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
# The three variants `ligand_parent_exact` exists to bridge.
#
# These tests are new because the old suite passed on a `parent_inchikey` that
# stripped salts and nothing else — so it caught the salt case above (the one it
# tested) and missed the two it did not. On the shipped KG the relation came out a
# 100 % duplicate of `ligand_exact` (6,939 == 6,939, identical pairs): it bridged
# ZERO of the variants it exists for, while 41,238 pairs of Ligand nodes that are the
# same compound sat in the graph with no edge between them at all.
#
# A test suite that only tests the case that works is how that survives.
# ---------------------------------------------------------------------------


def test_parent_inchikey_bridges_protonation():
    """A protonated amine and its neutral twin are the same compound.

    RDKit's SaltRemover deletes counterions; it does NOT neutralise a charge. The
    corpora disagree here because their loaders sanitise differently — see
    fixes.one_scaffold_per_ligand, which documents exactly this drift on N-oxides.
    """
    charged = "C[NH+]1CCN(c2ccccc2)CC1"
    neutral = "CN1CCN(c2ccccc2)CC1"
    assert vc.parent_inchikey(charged) == vc.parent_inchikey(neutral)
    # ...and they are NOT the same molecule to the full InChIKey, which is why the
    # bridge has to exist at all.
    assert vc.inchikey(charged) != vc.inchikey(neutral)


def test_parent_inchikey_bridges_stereo():
    """Two stereoisomers share a parent. ECFP4 is chirality-blind and already scores
    them as identical (weight 0.95); the identity relation must not contradict that."""
    r = "N[C@@H](C)C(=O)O"
    s = "N[C@H](C)C(=O)O"
    assert vc.parent_inchikey(r) == vc.parent_inchikey(s)
    assert vc.inchikey(r) != vc.inchikey(s)


def test_parent_inchikey_does_not_merge_different_compounds():
    """The bridge must not become a sledgehammer: unrelated molecules stay apart."""
    assert vc.parent_inchikey("CCO") != vc.parent_inchikey("CCC")
    # one methyl apart is still a different compound
    assert vc.parent_inchikey("CN1CCN(c2ccccc2)CC1") != \
           vc.parent_inchikey("CCN1CCN(c2ccccc2)CC1")


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
