"""Task 3 (stage A2, PyTorch): a neural network (MLP) on the stacked features.

A different family of nonlinear learner than gradient-boosted trees: smooth functions of
all features at once instead of axis-aligned splits, so its errors are partly uncorrelated
with LightGBM's and it adds diversity to the ensemble.

Protocol is identical to the LightGBM stack (regression_stack.py): fit on VALID with the SAME
5-fold split (KFold, shuffle, SEED), out-of-fold predictions on valid, fold-averaged on test/cold.
Inside each fold, 10% of the training rows are held out for early stopping, and every
preprocessing statistic (medians, scales) is fit on the fold's training rows only.

Never import LightGBM here (macOS OpenMP clash).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .config import ART, SEED
from . import metrics, splits
from .features import stack_features

torch.set_num_threads(8)
RES = ART / "results"


class Prep:
    """log-compress heavy-tailed columns, median-impute + missing indicators, standardize."""

    def fit(self, X: pd.DataFrame):
        A = X.values.astype(np.float64)
        self.heavy = np.nanmax(np.abs(A), 0) > 50
        A = self._log(A)
        self.med = np.nanmedian(A, 0)
        self.med = np.where(np.isnan(self.med), 0, self.med)
        self.miss = np.isnan(A).any(0)
        A = np.where(np.isnan(A), self.med, A)
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-6
        return self

    def _log(self, A):
        A = A.copy()
        A[:, self.heavy] = np.sign(A[:, self.heavy]) * np.log1p(np.abs(A[:, self.heavy]))
        return A

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        A = self._log(X.values.astype(np.float64))
        ind = np.isnan(A[:, self.miss]).astype(np.float32)
        A = (np.where(np.isnan(A), self.med, A) - self.mu) / self.sd
        return np.hstack([np.clip(A, -6, 6), ind]).astype(np.float32)


class MLP(torch.nn.Module):
    def __init__(self, d, hidden=(256, 128, 64), drop=0.1):
        super().__init__()
        layers, prev = [], d
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(drop)]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def fit_mlp(Xtr, ytr, Xes, yes, epochs=40, lr=1e-3, wd=1e-5, bs=2048, patience=4, seed=SEED, log=None):
    """Adam on MSE; early stopping on a held-out slice (Xes, yes). Returns the best model + history."""
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1])
    with torch.no_grad():
        m.net[-1].bias.fill_(float(ytr.mean()))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.float32))
    Xe = torch.from_numpy(Xes)
    g = torch.Generator().manual_seed(seed)
    best, state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(yt), generator=g)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = ((m(Xt[b]) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            v = metrics.mse(yes, m(Xe).numpy())
        sched.step(v)
        hist.append({"epoch": ep + 1, "train_mse": tot / len(yt), "es_mse": v})
        if v < best - 1e-4:
            best, state, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    return m, hist


@torch.no_grad()
def predict(m, X, bs=65536):
    return np.concatenate([m(torch.from_numpy(X[s:s + bs])).numpy() for s in range(0, len(X), bs)])


def run(log=print, folds=5):
    t0 = time.time()
    sets = {n: splits.load(n) for n in ["valid", "test", "cold"]}
    base = {n: pd.read_parquet(ART / f"reg_base_{n}.parquet") for n in sets}
    Xs = {n: stack_features(n, d, base[n]) for n, d in sets.items()}
    y = sets["valid"].y.values
    oof = np.zeros(len(y))
    others = {n: np.zeros(len(sets[n])) for n in ["test", "cold"]}
    rng = np.random.default_rng(SEED)
    gaps = []
    for f, (a, b) in enumerate(KFold(folds, shuffle=True, random_state=SEED).split(Xs["valid"])):
        es = rng.random(len(a)) < 0.1                       # early-stopping slice inside the training fold
        prep = Prep().fit(Xs["valid"].iloc[a[~es]])
        T = lambda X: prep.transform(X)
        Xa = T(Xs["valid"].iloc[a])
        m, hist = fit_mlp(Xa[~es], y[a][~es], Xa[es], y[a][es], seed=SEED + f)
        oof[b] = predict(m, T(Xs["valid"].iloc[b]))
        for n in others:
            others[n] += predict(m, T(Xs[n])) / folds
        tr_mse = metrics.mse(y[a][~es], predict(m, Xa[~es]))
        gaps.append({"fold": f, "epochs": len(hist), "train_mse": tr_mse, "oof_mse": metrics.mse(y[b], oof[b])})
        log(f"[3.7] MLP fold {f}: {len(hist)} epochs, train {tr_mse:.4f}  out-of-fold {gaps[-1]['oof_mse']:.4f} "
            f"({time.time() - t0:.0f}s)")
    log(f"[3.7] MLP OOF MSE {metrics.mse(y, oof):.4f}")
    for n, p in [("valid", oof), *others.items()]:
        pd.DataFrame({"MLP (neural net)": p.astype(np.float32)}).to_parquet(ART / f"reg_nn_{n}.parquet")
    with open(RES / "regression_nn.json", "w") as fh:
        json.dump({"folds": gaps}, fh, indent=1)
    log(f"[3.7] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
