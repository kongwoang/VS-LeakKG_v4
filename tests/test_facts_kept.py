"""The facts consolidate used to throw away, and the join that recovers one of them.

Two defects, one disease: a `DROPPED_*` entry justified by a mechanism that was never
built. `Split` was dropped "because the canonical schema emits partition assignments
separately" (there is no such code) and `LabelType` as "a static lookup table" (it is
the corpus's own assertion about each example — a DUD-E generated decoy, a LIT-PCBA
measured inactive and a BayesBind random molecule are all `label = 0`).
"""
from __future__ import annotations

import polars as pl
import pytest

from vsleakkg.kg import consolidate as C
from vsleakkg.kg import schema
from vsleakkg.load_bigbind import _assign_split


# --------------------------------------------------------------------------
# The facts survive consolidation
# --------------------------------------------------------------------------

def test_split_and_label_type_are_not_dropped():
    assert "Split" not in C.DROPPED_NODES
    assert "LabelType" not in C.DROPPED_NODES
    assert C.DROPPED_EDGES == frozenset()
    assert C.CORPUS_TO_CANONICAL_EDGE_TYPE["example_in_split"] == "example_in_split"
    assert C.CORPUS_TO_CANONICAL_EDGE_TYPE["example_has_label_type"] == "example_has_label_type"


def test_non_axis_edge_types_are_in_no_axis():
    """`lt:decoy` has degree 1.5 M. An axis that walked it would join every decoy to
    every other decoy at weight 1.00 x 1.00 and call the corpus wholly contaminated.
    That hazard is why these were deleted; keeping them out of the axes is the cure."""
    in_axis = {et for types in schema.AXIS_EDGE_TYPES.values() for et in types}
    assert not (schema.NON_AXIS_EDGE_TYPES & in_axis)


def test_every_non_axis_type_still_declares_a_weight():
    for et in schema.NON_AXIS_EDGE_TYPES:
        assert et in schema.DEFAULT_WEIGHTS


def _mini_graph(label_types, splits):
    n = len(label_types)
    nodes = pl.DataFrame({
        "node_id": [f"ex:S:T:{i}" for i in range(n)],
        "node_type": ["Example"] * n,
        "label": [""] * n,
        "props": ['{"label": 0, "source": "S"}'] * n,
    })
    edges = pl.DataFrame({
        "src": [f"ex:S:T:{i}" for i in range(n)] * 2,
        "dst": [f"lt:{t}" for t in label_types] + [f"split:{s}" for s in splits],
        "edge_type": ["example_has_label_type"] * n + ["example_in_split"] * n,
        "props": ["{}"] * (2 * n),
    })
    return nodes, edges


def test_enrich_writes_label_type_and_split_into_props():
    nodes, edges = _mini_graph(["decoy", "inactive", "random"],
                               ["DUD-E:unknown", "LIT-PCBA:train", "BayesBind:test"])
    out = C._enrich_example_props(nodes, edges)
    got = out.select(
        pl.col("props").str.json_path_match("$.label_type").alias("lt"),
        pl.col("props").str.json_path_match("$.split").alias("sp"),
        pl.col("props").str.json_path_match("$.label").alias("lb"),
    )
    assert got["lt"].to_list() == ["decoy", "inactive", "random"]
    assert got["sp"].to_list() == ["DUD-E:unknown", "LIT-PCBA:train", "BayesBind:test"]
    assert got["lb"].to_list() == ["0", "0", "0"], "the pre-existing props must survive"


def test_enrich_raises_rather_than_letting_a_null_through():
    """A null label_type reaching a contamination score is worse than a crash here."""
    nodes, edges = _mini_graph(["decoy", "inactive"], ["DUD-E:unknown", "LIT-PCBA:train"])
    edges = edges.filter(pl.col("edge_type") != "example_has_label_type")
    with pytest.raises(ValueError, match="label_type"):
        C._enrich_example_props(nodes, edges)


# --------------------------------------------------------------------------
# BigBind's published split
# --------------------------------------------------------------------------

_HDR = "lig_smiles,uniprot,active,pchembl_value\n"
_ROWS = ["CCO,P1,True,5.5\n", "CCN,P2,False,\n", "CCC,P1,True,6.0\n", "CCF,P3,False,\n"]


def _write_bigbind(tmp_path, split_of):
    (tmp_path / "activities_all.csv").write_text(_HDR + "".join(_ROWS))
    for name, fname in {"train": "activities_train.csv", "val": "activities_val.csv",
                        "test": "activities_test.csv"}.items():
        rows = [r for r, s in zip(_ROWS, split_of) if s == name]
        (tmp_path / fname).write_text(_HDR + "".join(rows))
    return tmp_path / "activities_all.csv"


def test_assign_split_recovers_the_published_split_and_keeps_row_order(tmp_path, caplog):
    """Row order is load-bearing: Example ids are `ex:<source>:<target>:<row_idx>`, so
    reading the three split CSVs and concatenating would renumber every BigBind
    Example. The split must be joined into activities_all's order, not replace it."""
    split_of = ["train", "test", "val", "train"]
    csv = _write_bigbind(tmp_path, split_of)
    df = pl.read_csv(csv, columns=["lig_smiles", "uniprot", "active", "pchembl_value"])
    out = _assign_split(df, tmp_path, csv, __import__("logging").getLogger("t"))

    assert out["split"].to_list() == split_of
    assert out["lig_smiles"].to_list() == ["CCO", "CCN", "CCC", "CCF"]
    assert "_rowkey" not in out.columns and "_ord" not in out.columns


def test_assign_split_joins_rows_whose_only_difference_is_a_null(tmp_path):
    """Joining on the parsed columns would lose these: polars does not match null to
    null, so every row with a null pchembl_value would come back with no split."""
    csv = _write_bigbind(tmp_path, ["train", "val", "train", "test"])
    df = pl.read_csv(csv, columns=["lig_smiles", "uniprot", "active", "pchembl_value"])
    out = _assign_split(df, tmp_path, csv, __import__("logging").getLogger("t"))
    null_rows = out.filter(pl.col("pchembl_value").is_null())
    assert null_rows.height == 2
    assert null_rows["split"].null_count() == 0


def test_assign_split_raises_when_the_files_do_not_partition(tmp_path):
    csv = _write_bigbind(tmp_path, ["train", "val", "train", "test"])
    # drop a row from test/ so one row of activities_all is in no split file
    (tmp_path / "activities_test.csv").write_text(_HDR)
    df = pl.read_csv(csv, columns=["lig_smiles", "uniprot", "active", "pchembl_value"])
    with pytest.raises(ValueError, match="in no split file"):
        _assign_split(df, tmp_path, csv, __import__("logging").getLogger("t"))


def test_assign_split_falls_back_loudly_when_the_files_are_absent(tmp_path):
    (tmp_path / "activities_all.csv").write_text(_HDR + "".join(_ROWS))
    csv = tmp_path / "activities_all.csv"
    df = pl.read_csv(csv, columns=["lig_smiles", "uniprot", "active", "pchembl_value"])
    out = _assign_split(df, tmp_path, csv, __import__("logging").getLogger("t"))
    assert out["split"].to_list() == ["unknown"] * 4
