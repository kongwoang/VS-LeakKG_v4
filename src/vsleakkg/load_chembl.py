"""Load ChEMBL via chembl-downloader (pinned version, default 35).

TODO:
- Use chembl_downloader.connect(version='35') for SQLite access.
- Pull activity records joined to molecule_dictionary + target_dictionary.
- Filter to relevant pchembl_value / standard_type / standard_units.
- Emit a Polars frame: (target_chembl_id, molecule_chembl_id, smiles, pchembl,
  standard_type, standard_units, assay_id, doc_id, source='ChEMBL').
"""
