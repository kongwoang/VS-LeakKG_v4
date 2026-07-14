"""BigBind V1.5 loader: activities -> Example/Ligand/Protein/Scaffold nodes.

The full BigBind archive (~18 GB) holds SDF/PDB structures, but the metadata
CSVs already contain everything we need for the KG:

  data/raw/BigBind/metadata/BigBindV1.5/
    activities_all.csv       583 K filtered activity rows (one per assay measurement)
    activities_unfiltered.csv  1.68 M raw activity rows
    activities_{train,test,val}.csv         standard split
    activities_sna_1_{train,test,val}.csv   SNA-balanced split
    structures_all.csv       19.9 K (UniProt, PDB, pocket center) triples
    structures_{train,test,val}.csv

We use `activities_all.csv` for KG construction (the curated, filtered set).
SNA splits and unfiltered are noisier; surface them via the `split_csv` arg.

Schema of activities_all.csv columns we consume:
  lig_smiles        SMILES (already canonical-ish, but we re-canonicalize via
                    RDKit so node IDs collapse with our other corpora)
  uniprot           UniProt accession -> Protein target ID
  active            bool -> label (1=active, 0=inactive)
  standard_type     IC50 / Ki / Kd / EC50 -> stored in props
  standard_value    nM -> stored in props
  pchembl_value     -log10(M) -> stored in props

Outputs the same shape as the other corpus loaders (load_dude / load_dekois /
etc.):

  build(meta_dir, ...) -> (examples_df, nodes_df, edges_df)

The (nodes, edges) frames are produced by the shared
`vsleakkg.build_graph.make_nodes_edges` builder so they slot directly into
the same `task_build_kg` concat path used by the other corpora.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from vsleakkg import chem as vc
from vsleakkg import build_graph as vb

_BIGBIND_ACTIVITIES_DEFAULT = "activities_all.csv"


def _featurize_batch(smiles: list[str], log: Optional[logging.Logger] = None
                     ) -> tuple[list[Optional[str]], list[Optional[str]], list[Optional[str]], list[bool]]:
    """Run RDKit canonical/scaffold/InChIKey over a list of SMILES, in parallel
    when possible.

    Uses `vsleakkg.chem.featurize_batch_parallel` which preserves input order
    and sanity-checks the result length + a few index alignments before
    returning. Returns parallel lists: (canonical, inchikey, scaffold, parse_ok).
    """
    feats = vc.featurize_batch_parallel(smiles, log=log)
    canon = [f.smiles_canonical for f in feats]
    iks = [f.inchikey for f in feats]
    scaf = [f.scaffold_smiles for f in feats]
    ok = [f.parse_ok for f in feats]
    return canon, iks, scaf, ok


def build(meta_dir: Path,
          extracted_dir: Optional[Path] = None,
          split_csv: str = _BIGBIND_ACTIVITIES_DEFAULT,
          log: Optional[logging.Logger] = None
          ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build (examples, nodes, edges) frames from BigBind metadata CSVs.

    Args:
        meta_dir: data/raw/BigBind/metadata/BigBindV1.5/
        extracted_dir: data/raw/BigBind/extracted/ (currently unused — 3D
            structures aren't needed for KG construction)
        split_csv: which activities CSV to consume. Defaults to the curated
            filtered set; pass `activities_unfiltered.csv` for the full
            ~1.68 M-row variant.
        log: optional logger
    """
    if log is None:
        log = logging.getLogger("vsleakkg.load_bigbind")
    csv_path = meta_dir / split_csv
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    log.info("BigBind: reading %s", csv_path)
    df = pl.read_csv(
        csv_path,
        columns=[
            "lig_smiles", "uniprot", "active",
            "standard_type", "standard_value", "standard_units",
            "pchembl_value", "ex_rec_pdb",
        ],
    )
    log.info("BigBind: %d activity rows", df.height)

    log.info("BigBind: re-canonicalizing SMILES via RDKit ...")
    canon, iks, scaffolds, ok_list = _featurize_batch(df["lig_smiles"].to_list(), log=log)
    df = df.with_columns([
        pl.Series("smiles_canonical", canon),
        pl.Series("inchikey", iks),
        pl.Series("scaffold_smiles", scaffolds),
        pl.Series("parse_ok", ok_list),
    ])
    bad = int((~df["parse_ok"]).sum())
    if bad:
        log.warning("BigBind: %d rows failed RDKit parse (excluded)", bad)
    df = df.filter(pl.col("parse_ok"))

    # Map to the loader contract.
    df = df.with_columns([
        pl.lit("BigBind").alias("source"),
        pl.col("uniprot").alias("target"),
        pl.when(pl.col("active")).then(1).otherwise(0).cast(pl.Int8).alias("label"),
        pl.when(pl.col("active")).then(pl.lit("active")).otherwise(pl.lit("inactive")).alias("label_type"),
        pl.lit("unknown").alias("split"),
        pl.col("ex_rec_pdb").alias("ext_id_1"),
        pl.lit(None, dtype=pl.Utf8).alias("ext_id_2"),
    ]).select([
        "smiles_canonical", "inchikey", "scaffold_smiles",
        "source", "target", "label", "label_type", "split",
        "ext_id_1", "ext_id_2",
        "standard_type", "standard_value", "standard_units", "pchembl_value",
    ])

    # Build the canonical (examples, nodes, edges).
    examples = vb.build_examples_frame(df)
    nodes, edges = vb.make_nodes_edges(
        examples,
        include_decoy_protocol=False,  # BigBind labels are measured, not protocol-decoy
        include_protein_target=True,
    )
    log.info("BigBind: %d examples, %d nodes, %d edges",
             examples.height, nodes.height, edges.height)
    return examples, nodes, edges
