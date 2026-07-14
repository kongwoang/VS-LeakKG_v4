"""The numpy Tanimoto kernel must agree with RDKit exactly — not approximately.

The ligand axis is the primary leakage axis. A kernel that is 6.5x faster and 0.1%
different is not a speedup, it is a silent corruption of every ligand-axis result.
So the property under test is equality of the emitted PAIR SET, not of a summary
statistic: same pairs, same scores, both directions of the set difference empty.
"""
from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from vsleakkg.ligand_similarity import _similar_chunk, _init_worker, pack_bitmatrix

# A spread of real drug-like scaffolds plus deliberate near-duplicates, so the
# window actually contains pairs above and below the threshold.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)OC", "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Nc1ccc(OC)cc1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "CN1C=NC2=C1C(=O)NC(=O)N2C",
    "c1ccc2c(c1)cccc2", "c1ccc2c(c1)ccc(c2)O", "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)OC",
    "COc1cc2c(cc1OC)CCNC2", "COc1cc2c(cc1O)CCNC2", "OC(=O)c1ccccc1O",
    "NC(=O)c1ccccc1O", "CCN(CC)CCNC(=O)c1ccc(N)cc1", "CCN(CC)CCNC(=O)c1ccc(NC)cc1",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCN(C)CC1", "CC1=C(C(=O)Nc2ccccc2)SC=N1",
    "CC1=C(C(=O)Nc2ccccc2C)SC=N1", "CCCCNC(=O)c1ccc(N)cc1", "O=C(O)Cc1ccccc1",
    "O=C(O)Cc1ccc(Cl)cc1", "NS(=O)(=O)c1ccc(cc1)C(=O)O",
]


def _rdkit_pairs(fps, pc, threshold, q_indices):
    """The original kernel, kept here as the reference implementation."""
    out = []
    for q in q_indices:
        hi = int(np.searchsorted(pc, int(pc[q] / threshold), side="right"))
        start = q + 1
        if start >= hi:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[q], fps[start:hi])
        for off, s in enumerate(sims):
            if s >= threshold:
                out.append((q, start + off, float(s)))
    return out


@pytest.fixture(scope="module")
def packed():
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
           for s in SMILES]
    blobs = [DataStructs.BitVectToBinaryText(f) for f in fps]
    pc = np.array([f.GetNumOnBits() for f in fps], dtype=np.int32)
    order = np.argsort(pc, kind="stable")
    pc = pc[order]
    fps = [fps[i] for i in order]
    blobs = [blobs[i] for i in order]
    mat = np.ascontiguousarray(pack_bitmatrix(blobs))
    return fps, mat, pc


def test_bitmatrix_popcounts_match_rdkit(packed):
    fps, mat, pc = packed
    assert (np.bitwise_count(mat).sum(axis=1).astype(np.int32) == pc).all()


@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.8, 0.85, 0.95])
def test_numpy_kernel_emits_exactly_the_rdkit_pairs(packed, threshold):
    fps, mat, pc = packed
    qs = list(range(len(pc)))
    _init_worker(pc, mat, [f"lig:{i}" for i in qs])

    got = _similar_chunk((qs, threshold))
    want = _rdkit_pairs(fps, pc, threshold, qs)

    assert {(q, t) for q, t, _ in got} == {(q, t) for q, t, _ in want}
    gd = {(q, t): s for q, t, s in got}
    wd = {(q, t): s for q, t, s in want}
    for k in wd:
        assert gd[k] == pytest.approx(wd[k], abs=1e-9), f"Tanimoto differs at {k}"


def test_the_bit_bound_never_prunes_a_real_pair(packed):
    """The pruning window is an optimisation. If it is wrong, edges vanish silently —
    so check the kernel against brute force over EVERY pair, with no window at all."""
    fps, mat, pc = packed
    threshold = 0.7
    n = len(pc)
    _init_worker(pc, mat, [f"lig:{i}" for i in range(n)])
    got = {(q, t) for q, t, _ in _similar_chunk((list(range(n)), threshold))}

    brute = set()
    for i in range(n):
        for j in range(i + 1, n):
            if DataStructs.TanimotoSimilarity(fps[i], fps[j]) >= threshold:
                brute.add((i, j))
    assert got == brute, f"window lost {brute - got}, invented {got - brute}"
