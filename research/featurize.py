"""Featurize every ligand as ECFP4 (Morgan r=2, 2048 bit, packed), then build an example
table (example -> ligand row, label, corpus, v4 partition). Feeds the ligand-based DL test.
Run: PYTHONNOUSERSITE=1 PYTHONPATH=src python research/featurize.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor

_gen = None


def fp_of(smi: str) -> np.ndarray:
    global _gen
    if _gen is None:
        from rdkit.Chem import rdFingerprintGenerator
        _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return np.zeros(256, np.uint8)
    return np.packbits(_gen.GetFingerprintAsNumPy(m).astype(np.uint8))


def main():
    nodes = pl.scan_parquet("outputs/kg/canonical_nodes.parquet")
    ligs = nodes.filter(pl.col("node_type") == "Ligand").select(pl.col("node_id").alias("lig"), pl.col("label").alias("smi")).collect()
    n = ligs.height
    print(f"featurizing {n:,} ligands ...", flush=True)
    smis = ligs["smi"].to_list()
    M = np.empty((n, 256), np.uint8)
    with ProcessPoolExecutor(max_workers=12) as ex:
        for i, fp in enumerate(ex.map(fp_of, smis, chunksize=4000)):
            M[i] = fp
            if i % 200000 == 0:
                print(f"  {i:,}/{n:,}", flush=True)
    np.save("research/lig_fp.npy", M)
    ligs.select("lig").with_row_index("lig_idx").write_parquet("research/lig_ids.parquet")
    print(f"saved lig_fp.npy {M.shape}", flush=True)

    # example table
    edges = pl.scan_parquet("outputs/kg/canonical_edges.parquet")
    ehl = (edges.filter(pl.col("edge_type") == "example_has_ligand")
           .select(pl.col("src").alias("ex"), pl.col("dst").alias("lig")).collect())
    lab_y = (nodes.filter(pl.col("node_type") == "Example")
             .select(pl.col("node_id").alias("ex"), pl.col("props").str.json_path_match("$.label").cast(pl.Int8).alias("y")).collect())
    v4 = pl.read_parquet("outputs/splits/vsleakkg_coldmulti_v4.parquet").rename({"example_id": "ex", "partition": "v4"})
    lig_idx = ligs.select("lig").with_row_index("lig_idx")
    tab = (ehl.join(lig_idx, on="lig").join(lab_y, on="ex").join(v4, on="ex")
           .with_columns(pl.col("ex").str.split(":").list.get(1).alias("corpus"))
           .select("ex", "lig_idx", "y", "corpus", "v4"))
    tab.write_parquet("research/example_features.parquet")
    print(f"saved example_features.parquet {tab.height:,} rows", flush=True)
    print(tab.group_by("v4").agg(pl.len(), pl.col("y").mean().round(3).alias("act")).sort("v4"))


if __name__ == "__main__":
    main()
