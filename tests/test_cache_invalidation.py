"""A cache that ignores the code is a machine for making fixes do nothing.

`_cached` used to be: if the three parquets exist, reuse them. So editing a loader
changed nothing until a human remembered to delete a file, and the pipeline reported
success either way. This is not hypothetical — it happened twice in this project:

  * The BigBind split fix silently did not run. A hard assertion on the expected
    counts caught it; a `log.warning` would not have.
  * The shipped KG mixed two RDKit versions across corpora, because BigBind and
    BayesBind were cached from a build under an older RDKit whose E/Z bond-direction
    convention differed. Ligand node ids are md5(canonical isomeric SMILES), so
    ~13,000 of them moved. The cache could not see it, because it was not looking.

The fingerprint therefore covers the loader source, the shared featurisation code,
and the RDKit version.
"""
from __future__ import annotations

import json

import pytest

from vsleakkg import build_kg


@pytest.fixture
def stamped(tmp_path, monkeypatch):
    """A slug whose three parquets exist and whose stamp is current."""
    monkeypatch.setattr(build_kg, "PROCESSED", tmp_path)
    for p in build_kg._cache_outputs("dude"):
        p.write_bytes(b"")
    build_kg.write_cache_stamp("dude")
    return tmp_path


def test_fresh_cache_is_reused(stamped, monkeypatch):
    monkeypatch.setattr(build_kg.pl, "read_parquet",
                        lambda *_a, **_k: type("F", (), {"height": 7})())
    assert build_kg._cached("dude") is not None


def test_missing_parquet_is_not_a_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(build_kg, "PROCESSED", tmp_path)
    build_kg.write_cache_stamp("dude")          # stamp but no parquets
    assert build_kg._cached("dude") is None


def test_an_unstamped_cache_is_rebuilt(stamped):
    """Every parquet built before this mechanism existed. None of them can be
    trusted, precisely because we cannot tell what built them."""
    (stamped / "dude_cache.json").unlink()
    assert build_kg._cached("dude") is None


def test_a_changed_loader_invalidates_the_cache(stamped):
    (stamped / "dude_cache.json").write_text(json.dumps({"code": "0" * 16}))
    assert build_kg._cached("dude") is None


def test_an_rdkit_upgrade_invalidates_the_cache(stamped, monkeypatch):
    """The defect this exists to prevent: node ids are md5(canonical SMILES), and
    the canonicaliser is RDKit. A different RDKit is a different graph."""
    import rdkit
    before = build_kg._code_fingerprint("dude")
    monkeypatch.setattr(rdkit, "__version__", rdkit.__version__ + ".pretend")
    after = build_kg._code_fingerprint("dude")
    assert before != after
    assert build_kg._cached("dude") is None     # the stamp was written under `before`


def test_every_cached_corpus_declares_its_loader():
    """A slug that reaches _cached without an entry here would raise KeyError at
    build time. Better to fail in a test than 40 minutes into a build."""
    assert set(build_kg._CACHE_LOADER) == {
        "dude", "dekois", "litpcba_ave", "bigbind", "bayesbind"}
    import pathlib
    src = pathlib.Path(build_kg.__file__).resolve().parent
    for name in list(build_kg._CACHE_LOADER.values()) + list(build_kg._CACHE_SHARED):
        assert (src / name).exists(), f"{name} is in the fingerprint but does not exist"
