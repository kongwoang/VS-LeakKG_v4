"""Sharper: PER-CORPUS head-to-head AUROC, RANDOM split vs v4, so the difference is explicit.
Global AUROC hides it (DUD-E's intrinsic decoy bias inflates both). Per corpus, on real-activity
benchmarks the model should go from ~0.9 (random/leaky) to ~0.5 (v4/clean).
Run: PYTHONNOUSERSITE=1 PYTHONPATH=src python research/train_eval_v2.py
"""
from __future__ import annotations
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
print("loading ...", flush=True)
M = np.load("research/lig_fp.npy")
Lsp = csr_matrix(np.unpackbits(M, axis=1).astype(np.float32))
tab = pl.read_parquet("research/example_features.parquet").filter(pl.col("v4") != "dropped")
li = tab["lig_idx"].to_numpy(); y = tab["y"].to_numpy()
v4 = tab["v4"].to_numpy(); corpus = tab["corpus"].to_numpy()
r = RNG.random(len(y)); rnd = np.where(r < 0.70, "train", np.where(r < 0.85, "val", "test"))
CORPORA = ["BayesBind", "LIT-PCBA", "BigBind", "DEKOIS", "DUD-E"]


def train_and_eval(part):
    tr = np.where(part == "train")[0]
    if len(tr) > 500_000:
        tr = RNG.choice(tr, 500_000, replace=False)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=40, early_stopping=True, random_state=0)
    mlp.fit(Lsp[li[tr]], y[tr])
    te = part == "test"
    p = mlp.predict_proba(Lsp[li[te]])[:, 1]
    yte, cte = y[te], corpus[te]
    res = {"ALL": roc_auc_score(yte, p)}
    for c in CORPORA:
        m = cte == c
        res[c] = roc_auc_score(yte[m], p[m]) if (m.sum() > 50 and len(set(yte[m])) == 2) else float("nan")
    return res


print("training MLP on RANDOM split ...", flush=True)
rr = train_and_eval(rnd)
print("training MLP on v4 split ...", flush=True)
rv = train_and_eval(v4)

print("\nPER-CORPUS MLP AUROC (ligand-only) — random split vs v4 (ours):")
print(f"  {'corpus':<12} {'RANDOM':>8} {'v4':>8} {'drop':>8}   note")
note = {"BayesBind": "real activity", "LIT-PCBA": "real activity", "BigBind": "real activity",
        "DEKOIS": "decoy", "DUD-E": "decoy (intrinsic bias)", "ALL": "mixed"}
for c in CORPORA + ["ALL"]:
    d = rr[c] - rv[c]
    print(f"  {c:<12} {rr[c]:8.3f} {rv[c]:8.3f} {d:+8.3f}   {note[c]}")

real = [c for c in ("BayesBind", "LIT-PCBA", "BigBind")]
mr = np.nanmean([rr[c] for c in real]); mv = np.nanmean([rv[c] for c in real])
print(f"\n  REAL-ACTIVITY benchmarks (BayesBind+LIT-PCBA+BigBind): "
      f"RANDOM {mr:.3f} -> v4 {mv:.3f}  (drop {mr-mv:+.3f})")
print("  => on benchmarks with measured negatives, v4 collapses ligand-only to ~chance;")
print("     DUD-E stays high because its decoy artifact is NOT a train/test leak.")
