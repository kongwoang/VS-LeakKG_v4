"""Minimal ChEMBL SQLite reader for VS-LeakKG provenance.

Pulls only what the audit needs:
  - molecule_dictionary: chembl_id, molregno, pref_name, max_phase
  - compound_structures: molregno, canonical_smiles, standard_inchi_key
  - activities: minimal cols, joined to molregno-of-interest only
  - assays / docs / target_dictionary: lookup tables
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

import polars as pl


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def list_tables(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )]


def table_schema(conn: sqlite3.Connection, table: str) -> List[tuple]:
    return list(conn.execute(f"PRAGMA table_info({table})"))


def count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def load_ligands(conn: sqlite3.Connection) -> pl.DataFrame:
    """One row per (molregno) with canonical SMILES, InChIKey, ChEMBL id."""
    q = """
    SELECT md.molregno, md.chembl_id AS molecule_chembl_id,
           md.pref_name, md.max_phase,
           cs.canonical_smiles, cs.standard_inchi_key
    FROM molecule_dictionary md
    LEFT JOIN compound_structures cs USING(molregno)
    """
    rows = list(conn.execute(q))
    return pl.DataFrame(rows, schema=[
        "molregno", "molecule_chembl_id", "pref_name", "max_phase",
        "canonical_smiles", "standard_inchi_key",
    ], orient="row")


def load_targets(conn: sqlite3.Connection) -> pl.DataFrame:
    q = """
    SELECT td.tid, td.chembl_id AS target_chembl_id, td.pref_name,
           td.target_type, td.organism
    FROM target_dictionary td
    """
    return pl.DataFrame(list(conn.execute(q)),
        schema=["tid", "target_chembl_id", "pref_name", "target_type", "organism"],
        orient="row")


def load_target_sequences(conn: sqlite3.Connection) -> pl.DataFrame:
    """One row per (tid, accession) with the UniProt accession + sequence.

    Joins target_dictionary -> target_components -> component_sequences. A
    single tid can have multiple components (e.g. heteromeric complexes);
    each row is one protein chain.
    """
    q = """
    SELECT td.tid, td.chembl_id AS target_chembl_id,
           cs.accession, cs.sequence, cs.component_type, cs.organism,
           cs.description
    FROM target_dictionary td
    JOIN target_components tc ON tc.tid = td.tid
    JOIN component_sequences cs ON cs.component_id = tc.component_id
    WHERE cs.sequence IS NOT NULL
      AND cs.component_type = 'PROTEIN'
    """
    return pl.DataFrame(list(conn.execute(q)),
        schema=["tid", "target_chembl_id", "accession", "sequence",
                "component_type", "organism", "description"],
        orient="row")


def load_documents(conn: sqlite3.Connection) -> pl.DataFrame:
    q = """
    SELECT d.doc_id, d.chembl_id AS document_chembl_id,
           d.pubmed_id, d.doi, d.year, d.journal, d.title
    FROM docs d
    """
    return pl.DataFrame(list(conn.execute(q)),
        schema=["doc_id", "document_chembl_id", "pubmed_id", "doi", "year", "journal", "title"],
        orient="row")


def load_assays(conn: sqlite3.Connection) -> pl.DataFrame:
    q = """
    SELECT a.assay_id, a.chembl_id AS assay_chembl_id, a.assay_type,
           a.description, a.doc_id, a.tid
    FROM assays a
    """
    return pl.DataFrame(list(conn.execute(q)),
        schema=["assay_id", "assay_chembl_id", "assay_type", "description", "doc_id", "tid"],
        orient="row")


def load_activities_for_molregnos(conn: sqlite3.Connection,
                                   molregnos: Iterable[int]) -> pl.DataFrame:
    """Pull minimal activity rows for the subset of molregnos we care about."""
    molregnos = list({int(m) for m in molregnos if m is not None})
    if not molregnos:
        return pl.DataFrame(schema={
            "activity_id": pl.Int64, "molregno": pl.Int64, "assay_id": pl.Int64,
            "standard_type": pl.Utf8, "standard_relation": pl.Utf8,
            "standard_value": pl.Float64, "standard_units": pl.Utf8,
            "pchembl_value": pl.Float64, "doc_id": pl.Int64,
        })
    # SQLite has a 999-parameter default; we batch with a temp table.
    rows: List[tuple] = []
    cur = conn.cursor()
    cur.execute("CREATE TEMP TABLE IF NOT EXISTS _mol_subset (molregno INTEGER PRIMARY KEY)")
    cur.execute("DELETE FROM _mol_subset")
    cur.executemany("INSERT OR IGNORE INTO _mol_subset(molregno) VALUES (?)",
                    [(m,) for m in molregnos])
    q = """
    SELECT a.activity_id, a.molregno, a.assay_id,
           a.standard_type, a.standard_relation, a.standard_value,
           a.standard_units, a.pchembl_value, a.doc_id
    FROM activities a
    JOIN _mol_subset s ON a.molregno = s.molregno
    """
    rows = list(cur.execute(q))
    return pl.DataFrame(rows, schema=[
        "activity_id", "molregno", "assay_id",
        "standard_type", "standard_relation", "standard_value",
        "standard_units", "pchembl_value", "doc_id",
    ], orient="row")
