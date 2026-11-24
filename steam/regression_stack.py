"""Task 3 (stage B, LightGBM): stacked gradient boosting, hurdle model, ensemble.

Reads stage-A predictions (regression_cf.py, fit on TRAIN) and fits on VALID with
5-fold cross-fitting: out-of-fold predictions on valid, fold-averaged on
test/cold. No model ever sees its own target. Stage 1 of the hurdle model is
the Task 4 "will they play?" classifier.

Never import torch in this module (macOS OpenMP clash, see regression_cf.py).
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .config import ART, SEED
from . import features, metrics, splits

RES = ART / "results"
RES.mkdir(exist_ok=True)

LGB_REG = dict(objective="regression", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
               feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
               verbose=-1, seed=SEED, num_threads=8)
LGB_CLF = {**LGB_REG, "objective": "binary"}


def crossfit_lgb(X_va, y_va, X_others, params, n_rounds=1000, folds=5):
    """K-fold cross-fitting on VALID: OOF predictions for valid, fold-average for others."""
    oof = np.zeros(len(X_va))
    others = [np.zeros(len(X)) for X in X_others]
    models = []
    for a, b in KFold(folds, shuffle=True, random_state=SEED).split(X_va):
        bst = lgb.train(params, lgb.Dataset(X_va.iloc[a], y_va[a]), n_rounds,
                        valid_sets=[lgb.Dataset(X_va.iloc[b], y_va[b])],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = bst.predict(X_va.iloc[b], num_iteration=bst.best_iteration)
        for o, X in zip(others, X_others):
            o += bst.predict(X, num_iteration=bst.best_iteration) / folds
        models.append(bst)
    return oof, others, models


def stack_features(name: str, d: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Leak-free pair features + stage-A model outputs."""
    X = features.pair_features(d)
    X["bias_pred"] = base["Bias model"].values
    X["mf_pred"] = base["Latent factor (MF)"].values
    X["mf_dot"] = base["_mf_dot"].values
    X["fm_pred"] = base["Factorization machine"].values
    return X


def run(log=print):
    t0 = time.time()
    sets = {n: splits.load(n) for n in ["valid", "test", "cold"]}
    va = sets["valid"]
    preds = {n: pd.read_parquet(ART / f"reg_base_{n}.parquet") for n in sets}
    Xs = {n: stack_features(n, d, preds[n]) for n, d in sets.items()}
    log(f"[3.4] features: {Xs['valid'].shape[1]} columns ({time.time() - t0:.0f}s)")

    # ---- 3.4 LightGBM on stacked features
    oof, (p_te, p_co), models = crossfit_lgb(Xs["valid"], va.y.values, [Xs["test"], Xs["cold"]], LGB_REG)
    for n, p in zip(sets, [oof, p_te, p_co]):
        preds[n]["LightGBM (stacked)"] = p
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], 0), index=Xs["valid"].columns)
    imp.sort_values(ascending=False).to_json(RES / "lgb_importance.json")
    log(f"[3.4] LightGBM OOF MSE {metrics.mse(va.y.values, oof):.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.4b cold-safe LightGBM: no game-identity signal (no item stats, no CF outputs).
    # The only model that can legitimately score never-seen games; also the
    # "remove collaborative features" ablation on warm games.
    meta_cols = [c for c in Xs["valid"].columns if c.startswith(("user_", "meta_", "ug_"))]
    oof_m, (te_m, co_m), _ = crossfit_lgb(Xs["valid"][meta_cols], va.y.values,
                                          [Xs["test"][meta_cols], Xs["cold"][meta_cols]], LGB_REG)
    for n, p in zip(sets, [oof_m, te_m, co_m]):
        preds[n]["LightGBM (metadata only)"] = p
    log(f"[3.4b] metadata-only LightGBM OOF MSE {metrics.mse(va.y.values, oof_m):.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.5 hurdle: P(played) x E[y | played]
    # MSE-optimal prediction is E[y] = P(played) * E[y | played] because y = 0 when unplayed.
    oof_p, (te_p, co_p), clf_models = crossfit_lgb(Xs["valid"], va.played.values, [Xs["test"], Xs["cold"]], LGB_CLF)
    pl = va.played.values == 1
    oof_mu, (va_mu, te_mu, co_mu), _ = crossfit_lgb(Xs["valid"][pl].reset_index(drop=True), va.y.values[pl],
                                                  [Xs["valid"], Xs["test"], Xs["cold"]], LGB_REG)
    va_mu = va_mu.copy()
    va_mu[pl] = oof_mu                       # out-of-fold wherever the row trained stage 2
    for n, p, mu in zip(sets, [oof_p, te_p, co_p], [va_mu, te_mu, co_mu]):
        preds[n]["Hurdle (LightGBM)"] = p * mu
    np.savez(ART / "played_probs_lgb.npz", valid=oof_p, test=te_p, cold=co_p)
    log(f"[3.5] hurdle OOF MSE {metrics.mse(va.y.values, preds['valid']['Hurdle (LightGBM)']):.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.6 ensemble: linear blend fit on valid (OOF for the stacked models)
    base = ["Bias model", "Latent factor (MF)", "Factorization machine", "LightGBM (stacked)", "Hurdle (LightGBM)"]
    blend = LinearRegression().fit(preds["valid"][base].values, va.y.values)
    for n in sets:
        preds[n]["Ensemble"] = blend.predict(preds[n][base].values)
    tuning = {"blend_weights": dict(zip(base, blend.coef_.tolist())),
              "blend_intercept": float(blend.intercept_)}
    log(f"[3.6] blend weights {tuning['blend_weights']}")

    results = {}
    for n, d in sets.items():
        P = preds[n].drop(columns=[c for c in preds[n].columns if c.startswith("_")])
        results[n] = {m: metrics.regression_report(d.y.values, P[m].values) for m in P.columns}
        if n != "valid":
            P.astype(np.float32).to_parquet(ART / f"reg_preds_{n}.parquet")
    with open(RES / "regression.json", "w") as f:
        json.dump({"results": results, "tuning": tuning}, f, indent=1)
    log(f"[3B] done in {time.time() - t0:.0f}s")
    return results


def breakdown(name="test") -> pd.DataFrame:
    """MSE per model by user-activity and game-popularity bins (F18)."""
    d = splits.load(name)
    P = pd.read_parquet(ART / f"reg_preds_{name}.parquet")
    ustats, istats, _, _ = features.train_stats()
    ub = pd.cut(ustats.n_train.values[d.u.values], [-1, 10, 50, 200, 1e9],
                labels=["≤10", "11–50", "51–200", ">200"])
    ib = pd.cut(istats.n_train.values[d.i.values], [-1, 100, 1000, 10000, 1e9],
                labels=["≤100", "101–1k", "1k–10k", ">10k"])
    rows = []
    for col in P.columns:
        err = (P[col].values - d.y.values) ** 2
        for kind, b in [("user's train games", ub), ("game's train owners", ib)]:
            for k, v in pd.Series(err).groupby(pd.Series(b), observed=True).mean().items():
                rows.append({"model": col, "by": kind, "bin": k, "MSE": v})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
