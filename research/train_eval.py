"""Ligand-based DL test: does a model that predicts active/inactive from the ligand alone
generalise, or does it memorise? Train on train, eval AUROC on test, for the v4 cold-multi-axis
split vs a RANDOM split over the SAME retained pool. If v4 is genuinely harder, AUROC collapses
toward 0.5 under v4 while staying high under random — i.e. legacy performance was ligand memorisation.
Run: PYTHONNOUSERSITE=1 PYTHONPATH=src python research/train_eval.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
print("loading features ...", flush=True)
M = np.load("research/lig_fp.npy")                       # [n_lig, 256] packed
Lsp = csr_matrix(np.unpackbits(M, axis=1).astype(np.float32))  # [n_lig, 2048] sparse
tab = pl.read_parquet("research/example_features.parquet")
pool = tab.filter(pl.col("v4") != "dropped")             # same example pool for both splits
li = pool["lig_idx"].to_numpy()
y = pool["y"].to_numpy()
v4 = pool["v4"].to_numpy()
corpus = pool["corpus"].to_numpy()
# random split over the SAME pool, 70/15/15
r = RNG.random(len(y)); rnd = np.where(r < 0.70, "train", np.where(r < 0.85, "val", "test"))
print(f"pool {len(y):,} | active {y.mean():.3f}", flush=True)


def evaluate(part, label):
    tr = part == "train"; te = part == "test"
    Xtr, ytr = Lsp[li[tr]], y[tr]
    Xte, yte = Lsp[li[te]], y[te]
    # subsample train for the MLP (keep it fast); logistic uses full train
    idx = np.arange(Xtr.shape[0])
    if len(idx) > 500_000:
        idx = RNG.choice(idx, 500_000, replace=False)
    out = {"n_test": int(te.sum()), "test_act": float(yte.mean())}
    lr = LogisticRegression(max_iter=200, C=1.0, solver="saga", n_jobs=8)
    lr.fit(Xtr, ytr)
    out["logreg"] = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=40, early_stopping=True, random_state=0)
    mlp.fit(Xtr[idx], ytr[idx])
    out["mlp"] = roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1])
    print(f"  {label:<10} test n={out['n_test']:>8,} act={out['test_act']:.2f} | "
          f"LogReg AUROC={out['logreg']:.3f} | MLP AUROC={out['mlp']:.3f}", flush=True)
    return out, (yte, mlp.predict_proba(Xte)[:, 1], corpus[te])


print("\nLIGAND-ONLY generalisation (higher AUROC = more memorisable / leakier split):")
r_rand, _ = evaluate(rnd, "RANDOM")
r_v4, (yte, pte, cte) = evaluate(v4, "v4 (ours)")
print(f"\n  => AUROC drop v4 vs random: LogReg {r_rand['logreg']-r_v4['logreg']:+.3f} | "
      f"MLP {r_rand['mlp']-r_v4['mlp']:+.3f}  (big drop = v4 removes ligand memorisation)")

print("\nper-corpus MLP AUROC on v4 test:")
for c in sorted(set(cte.tolist())):
    m = cte == c
    if m.sum() > 50 and len(set(yte[m].tolist())) == 2:
        print(f"  {c:<12} n={int(m.sum()):>7,} AUROC={roc_auc_score(yte[m], pte[m]):.3f}")
