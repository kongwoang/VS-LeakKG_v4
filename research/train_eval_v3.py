"""The comparison that matters: ligand-only AUROC under the splits people ACTUALLY use vs v4.
Random is a trivial baseline; the real question is whether v4 is harder than the published
leakage-controlled splits (BigBind/LIT-PCBA/BayesBind own splits) and than a standard 1-axis
scaffold split. Per corpus, same model, train subsampled to a fixed size for fairness.
Run: PYTHONNOUSERSITE=1 PYTHONPATH=src python research/train_eval_v3.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
NTRAIN = 60_000
print("loading ...", flush=True)
M = np.load("research/lig_fp.npy")
Lsp = csr_matrix(np.unpackbits(M, axis=1).astype(np.float32))
nodes = pl.scan_parquet("outputs/kg/canonical_nodes.parquet")
edges = pl.scan_parquet("outputs/kg/canonical_edges.parquet")
tab = pl.read_parquet("research/example_features.parquet")  # ex, lig_idx, y, corpus, v4
# published partition
eis = (edges.filter(pl.col("edge_type") == "example_in_split")
       .select(pl.col("src").alias("ex"), pl.col("dst").str.split(":").list.get(2).alias("pub")).collect())
# scaffold per example (via its ligand's Murcko scaffold)
ehl = edges.filter(pl.col("edge_type") == "example_has_ligand").select(pl.col("src").alias("ex"), pl.col("dst").alias("lig")).collect()
lsc = edges.filter(pl.col("edge_type") == "ligand_scaffold").select(pl.col("src").alias("lig"), pl.col("dst").alias("sca")).collect()
ex_sca = ehl.join(lsc, on="lig").select("ex", "sca")
tab = tab.join(eis, on="ex", how="left").join(ex_sca, on="ex", how="left")

PUB = {"BigBind": ("train", "test"), "LIT-PCBA": ("train", "validation"), "BayesBind": ("val", "test")}


def fit_auroc(df, tr_mask, te_mask):
    tri = np.where(tr_mask)[0]
    if len(tri) < 200 or te_mask.sum() < 50:
        return float("nan")
    if len(tri) > NTRAIN:
        tri = RNG.choice(tri, NTRAIN, replace=False)
    li = df["lig_idx"].to_numpy(); y = df["y"].to_numpy()
    if len(set(y[tri])) < 2 or len(set(y[te_mask])) < 2:
        return float("nan")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=40, early_stopping=True, random_state=0)
    mlp.fit(Lsp[li[tri]], y[tri])
    p = mlp.predict_proba(Lsp[li[te_mask]])[:, 1]
    return roc_auc_score(y[te_mask], p)


print(f"\nLIGAND-ONLY MLP AUROC per corpus (train subsampled to {NTRAIN:,}):")
print(f"  {'corpus':<11} {'random':>8} {'published':>10} {'scaffold':>9} {'v4 (ours)':>10}")
for corpus in ["BigBind", "LIT-PCBA", "BayesBind", "DUD-E", "DEKOIS"]:
    df = tab.filter(pl.col("corpus") == corpus)
    n = df.height
    pub = df["pub"].to_numpy(); v4 = df["v4"].to_numpy(); sca = df["sca"].to_numpy()
    # random 80/20
    rr = RNG.random(n); rnd_tr, rnd_te = rr < 0.8, rr >= 0.8
    # scaffold 80/20 (assign each scaffold to train/test)
    uniq = pl.Series(sca).unique().to_list()
    assign = {s: (RNG.random() < 0.8) for s in uniq}
    sca_tr = np.array([assign.get(s, True) for s in sca])
    a_rand = fit_auroc(df, rnd_tr, rnd_te)
    a_sca = fit_auroc(df, sca_tr, ~sca_tr)
    a_v4 = fit_auroc(df, v4 == "train", v4 == "test")
    if corpus in PUB:
        tr, te = PUB[corpus]
        a_pub = fit_auroc(df, pub == tr, pub == te)
    else:
        a_pub = float("nan")  # DUD-E/DEKOIS: no published train/test split
    print(f"  {corpus:<11} {a_rand:8.3f} {a_pub:10.3f} {a_sca:9.3f} {a_v4:10.3f}", flush=True)

print("\n(published/scaffold = existing leakage-control; v4 lower = harder than what people use.")
print(" DUD-E/DEKOIS: no published split; high everywhere = intrinsic decoy bias, not leakage.)")
