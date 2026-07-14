"""Load BindingDB TSV dump.

TODO:
- Stream-read BindingDB_All.tsv (zip or extracted) with Polars/pyarrow.
- Normalize column subset: ligand_smiles, target_name, target_uniprot, ki/kd/ic50,
  pmid, doi.
- Emit a Polars frame with provenance fields and unit-normalized affinity.
"""
