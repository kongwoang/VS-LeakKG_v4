"""Chemistry primitives: SMILES canonicalization, InChIKey, ECFP4, Bemis-Murcko,
Tanimoto. RDKit is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

_SALT_REMOVER = SaltRemover.SaltRemover()

# RDKit emits warnings for sub-MOL2 parse failures and odd valences. Silence them
# for batch processing; callers that need to inspect individual molecules can
# re-enable temporarily.
RDLogger.DisableLog("rdApp.*")

ECFP_RADIUS = 2
ECFP_NBITS = 2048


@dataclass(slots=True)
class MolFeatures:
    """Per-ligand summary produced by `featurize`."""
    smiles_input: str
    smiles_canonical: Optional[str]
    inchikey: Optional[str]
    scaffold_smiles: Optional[str]
    parse_ok: bool


def _parse(smi: str) -> Optional[Chem.Mol]:
    if smi is None:
        return None
    smi = smi.strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return mol


def canonicalize_smiles(smi: str, isomeric: bool = True) -> Optional[str]:
    mol = _parse(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def inchikey(smi: str) -> Optional[str]:
    mol = _parse(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def bemis_murcko_scaffold(smi: str) -> Optional[str]:
    """Generic Bemis-Murcko scaffold as a canonical SMILES. Empty scaffolds
    (e.g. fully aliphatic acyclic mols) are reported as the empty string."""
    mol = _parse(smi)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def ecfp(smi: str, radius: int = ECFP_RADIUS, nbits: int = ECFP_NBITS):
    """RDKit ExplicitBitVect (used by BulkTanimotoSimilarity)."""
    mol = _parse(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def ecfp_bytes(smi: str, radius: int = ECFP_RADIUS, nbits: int = ECFP_NBITS) -> Optional[bytes]:
    """Bit-packed ECFP4 suitable for parquet storage and numpy reload."""
    fp = ecfp(smi, radius, nbits)
    if fp is None:
        return None
    return DataStructs.BitVectToBinaryText(fp)


def bytes_to_fp(b: bytes, nbits: int = ECFP_NBITS):
    # CreateFromBinaryText is a module-level factory; pass the binary blob from
    # BitVectToBinaryText. Bytes are auto-converted to std::string by the C++
    # binding. The `nbits` arg is kept only for API symmetry — RDKit infers it
    # from the binary header.
    return DataStructs.CreateFromBinaryText(b)


def featurize(smi: str) -> MolFeatures:
    """One-shot canonical + scaffold + inchikey. Avoids parsing the same SMILES
    multiple times."""
    mol = _parse(smi)
    if mol is None:
        return MolFeatures(smi, None, None, None, False)
    can = Chem.MolToSmiles(mol, isomericSmiles=True)
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        ik = None
    try:
        scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        scaf = None
    return MolFeatures(smi, can, ik, scaf, True)


def featurize_batch_parallel(smiles: list[str], n_workers: Optional[int] = None,
                              chunksize: int = 1000,
                              log: Optional["logging.Logger"] = None
                              ) -> list[MolFeatures]:
    """Featurize a list of SMILES in parallel using multiprocessing.

    **Order preservation**: uses `pool.imap` (NOT imap_unordered) so results
    arrive in the same index order as input. A length sanity check is run at
    the end; if it fails, we raise rather than return partial data.

    Falls back to sequential featurize on a single worker or when the list is
    smaller than chunksize.
    """
    import logging as _logging
    import multiprocessing as _mp
    import os
    if log is None:
        log = _logging.getLogger("vsleakkg.chem.featurize_batch_parallel")
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)
    if n_workers <= 1 or len(smiles) <= chunksize:
        return [featurize(s) for s in smiles]
    log.info("featurize parallel: %d SMILES on %d workers, chunksize=%d",
             len(smiles), n_workers, chunksize)
    with _mp.get_context("spawn").Pool(n_workers) as pool:
        # imap preserves order; we still verify length + sentinel before returning.
        out = list(pool.imap(featurize, smiles, chunksize=chunksize))
    if len(out) != len(smiles):
        raise RuntimeError(
            f"featurize_batch_parallel returned {len(out)} items for {len(smiles)} input")
    # Spot-check first and last entry: smiles_input must round-trip.
    if smiles and out:
        for idx in (0, len(smiles) // 2, len(smiles) - 1):
            if smiles[idx] != out[idx].smiles_input:
                raise RuntimeError(
                    f"featurize parallel order mismatch at index {idx}: "
                    f"input={smiles[idx]!r} got={out[idx].smiles_input!r}")
    return out


def parent_inchikey_batch_parallel(smiles: list[str], n_workers: Optional[int] = None,
                                    chunksize: int = 2000,
                                    log: Optional["logging.Logger"] = None
                                    ) -> list[Optional[str]]:
    """Compute salt-stripped parent InChIKey for many SMILES in parallel.

    Same order-preservation contract as `featurize_batch_parallel`: results
    align by index with input. The returned list always has length
    `len(smiles)`; None entries mark parse / strip failures.
    """
    import logging as _logging
    import multiprocessing as _mp
    import os
    if log is None:
        log = _logging.getLogger("vsleakkg.chem.parent_inchikey_batch_parallel")
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)
    if n_workers <= 1 or len(smiles) <= chunksize:
        return [parent_inchikey(s) for s in smiles]
    log.info("parent_inchikey parallel: %d SMILES on %d workers, chunksize=%d",
             len(smiles), n_workers, chunksize)
    with _mp.get_context("spawn").Pool(n_workers) as pool:
        out = list(pool.imap(parent_inchikey, smiles, chunksize=chunksize))
    if len(out) != len(smiles):
        raise RuntimeError(
            f"parent_inchikey_batch_parallel returned {len(out)} items "
            f"for {len(smiles)} input")
    return out


def parent_inchikey(smi: str) -> Optional[str]:
    """Return the InChIKey of the salt-stripped parent molecule.

    Removes common counterions (Cl-, Na+, etc. via RDKit's default SaltRemover
    table) BEFORE computing the InChIKey. Two molecules that share the same
    parent skeleton but differ only by salt form / protonation will have the
    same `parent_inchikey` but different full `inchikey`. The KG uses this to
    bridge them via `same_parent_inchikey_as` edges.

    Returns None on parse failure or if salt-stripping leaves an empty
    fragment.
    """
    mol = _parse(smi)
    if mol is None:
        return None
    try:
        stripped = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        stripped = mol
    if stripped is None or stripped.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToInchiKey(stripped)
    except Exception:
        return None


def tanimoto(a, b) -> float:
    return DataStructs.TanimotoSimilarity(a, b)


def bulk_tanimoto(query, refs: Sequence) -> np.ndarray:
    """Vectorized Tanimoto from one query against a list of refs."""
    return np.asarray(DataStructs.BulkTanimotoSimilarity(query, list(refs)), dtype=np.float32)


def max_tanimoto_to_set(queries: Sequence, refs: Sequence) -> np.ndarray:
    """For each query, return its max Tanimoto similarity to any ref. None
    fingerprints are skipped (those rows get -1.0)."""
    out = np.full(len(queries), -1.0, dtype=np.float32)
    if not refs:
        return out
    ref_list = list(refs)
    for i, q in enumerate(queries):
        if q is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_list)
        out[i] = max(sims) if sims else -1.0
    return out


def count_pairs_above(queries: Sequence, refs: Sequence, thresholds: Iterable[float]) -> dict:
    """Count (query, ref) pairs whose Tanimoto >= each threshold. Uses bulk
    Tanimoto per query; thresholds is iterated cheaply on each row."""
    ths = sorted(set(thresholds))
    counts = {t: 0 for t in ths}
    if not queries or not refs:
        return counts
    ref_list = list(refs)
    for q in queries:
        if q is None:
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(q, ref_list), dtype=np.float32)
        for t in ths:
            counts[t] += int((sims >= t).sum())
    return counts
