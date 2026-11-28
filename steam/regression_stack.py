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
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.model_selection import KFold

from .config import ART, SEED
from . import features, metrics, splits
from .features import stack_features

RES = ART / "results"
RES.mkdir(exist_ok=True)

# Chosen by scripts/tune_lgb.py on VALID out-of-fold MSE ("reg_B"): best OOF for both the stacked and the
# metadata-only model, and a smaller train/OOF gap than the V2 setting (lr .05, 127 leaves, min 100, L2 1).
LGB_REG = dict(objective="regression", learning_rate=0.03, num_leaves=127, min_data_in_leaf=500,
               feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0,
               verbose=-1, seed=SEED, num_threads=8)
LGB_CLF = {**LGB_REG, "objective": "binary"}


def _es_split(a, frac=0.1, seed=SEED):
    """Early-stopping slice carved out of the TRAINING fold, so the out-of-fold rows never
    choose the number of trees (choosing it on the scored fold would leak and flatter OOF MSE)."""
    es = np.random.default_rng(seed).random(len(a)) < frac
    return a[~es], a[es]


FIT_GAP = {}       # model name -> mean in-fold train MSE vs out-of-fold MSE (overfitting check)


def crossfit_lgb(X_va, y_va, X_others, params, n_rounds=2000, folds=5, name=None):
    """K-fold cross-fitting on VALID: OOF predictions for valid, fold-average for others."""
    oof = np.zeros(len(X_va))
    others = [np.zeros(len(X)) for X in X_others]
    models, tr_err = [], []
    for a, b in KFold(folds, shuffle=True, random_state=SEED).split(X_va):
        fit, es = _es_split(a)
        bst = lgb.train(params, lgb.Dataset(X_va.iloc[fit], y_va[fit]), n_rounds,
                        valid_sets=[lgb.Dataset(X_va.iloc[es], y_va[es])],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = bst.predict(X_va.iloc[b], num_iteration=bst.best_iteration)
        for o, X in zip(others, X_others):
            o += bst.predict(X, num_iteration=bst.best_iteration) / folds
        models.append(bst)
        if name and params.get("objective") == "regression":
            tr_err.append(metrics.mse(y_va[fit], bst.predict(X_va.iloc[fit], num_iteration=bst.best_iteration)))
    if tr_err:
        FIT_GAP[name] = {"train_mse": float(np.mean(tr_err)), "oof_mse": metrics.mse(y_va, oof),
                         "rounds": float(np.mean([m.best_iteration for m in models]))}
    return oof, others, models


def crossfit_catboost(X_va, y_va, X_others, folds=5, name="CatBoost"):
    """Same protocol with CatBoost (ordered boosting on symmetric trees): a different tree learner."""
    oof = np.zeros(len(X_va))
    others = [np.zeros(len(X)) for X in X_others]
    tr_err, its = [], []
    for a, b in KFold(folds, shuffle=True, random_state=SEED).split(X_va):
        fit, es = _es_split(a)
        m = CatBoostRegressor(iterations=3000, learning_rate=0.08, depth=8, l2_leaf_reg=3, loss_function="RMSE",
                              random_seed=SEED, thread_count=8, verbose=0, od_type="Iter", od_wait=100)
        m.fit(X_va.iloc[fit], y_va[fit], eval_set=(X_va.iloc[es], y_va[es]), use_best_model=True)
        oof[b] = m.predict(X_va.iloc[b])
        for o, X in zip(others, X_others):
            o += m.predict(X) / folds
        tr_err.append(metrics.mse(y_va[fit], m.predict(X_va.iloc[fit])))
        its.append(m.get_best_iteration())
    FIT_GAP[name] = {"train_mse": float(np.mean(tr_err)), "oof_mse": metrics.mse(y_va, oof), "rounds": float(np.mean(its))}
    return oof, others


def run(log=print):
    t0 = time.time()
    sets = {n: splits.load(n) for n in ["valid", "test", "cold"]}
    va = sets["valid"]
    preds = {n: pd.read_parquet(ART / f"reg_base_{n}.parquet") for n in sets}
    Xs = {n: stack_features(n, d, preds[n]) for n, d in sets.items()}
    log(f"[3.4] features: {Xs['valid'].shape[1]} columns ({time.time() - t0:.0f}s)")

    # ---- 3.4 LightGBM on stacked features
    oof, (p_te, p_co), models = crossfit_lgb(Xs["valid"], va.y.values, [Xs["test"], Xs["cold"]], LGB_REG,
                                             name="LightGBM (stacked)")
    for n, p in zip(sets, [oof, p_te, p_co]):
        preds[n]["LightGBM (stacked)"] = p
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], 0), index=Xs["valid"].columns)
    imp.sort_values(ascending=False).to_json(RES / "lgb_importance.json")
    log(f"[3.4] LightGBM OOF MSE {metrics.mse(va.y.values, oof):.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.4b cold-safe LightGBM: no game-identity signal (no item stats, no CF outputs).
    # The only model that can legitimately score never-seen games; also the
    # "remove collaborative features" ablation on warm games.
    meta_cols = [c for c in Xs["valid"].columns if c.startswith(features.META_PREFIXES)]
    oof_m, (te_m, co_m), _ = crossfit_lgb(Xs["valid"][meta_cols], va.y.values,
                                          [Xs["test"][meta_cols], Xs["cold"][meta_cols]], LGB_REG,
                                          name="LightGBM (metadata only)")
    for n, p in zip(sets, [oof_m, te_m, co_m]):
        preds[n]["LightGBM (metadata only)"] = p
    log(f"[3.4b] metadata-only LightGBM OOF MSE {metrics.mse(va.y.values, oof_m):.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.5 hurdle: P(played) x E[y | played]
    # MSE-optimal prediction is E[y] = P(played) * E[y | played] because y = 0 when unplayed.
    oof_p, (te_p, co_p), clf_models = crossfit_lgb(Xs["valid"], va.played.values, [Xs["test"], Xs["cold"]], LGB_CLF)
    pl = va.played.values == 1
    oof_mu, (va_mu, te_mu, co_mu), _ = crossfit_lgb(Xs["valid"][pl].reset_index(drop=True), va.y.values[pl],
                                                  [Xs["valid"], Xs["test"], Xs["cold"]], LGB_REG,
                                                  name="Hurdle stage 2 (played only)")
    va_mu = va_mu.copy()
    va_mu[pl] = oof_mu                       # out-of-fold wherever the row trained stage 2
    for n, p, mu in zip(sets, [oof_p, te_p, co_p], [va_mu, te_mu, co_mu]):
        preds[n]["Hurdle (LightGBM)"] = p * mu
    np.savez(ART / "played_probs_lgb.npz", valid=oof_p, test=te_p, cold=co_p)
    log(f"[3.5] hurdle OOF MSE {metrics.mse(va.y.values, preds['valid']['Hurdle (LightGBM)']):.4f} ({time.time() - t0:.0f}s)")

    # ---- 3.7 other nonlinear learners: CatBoost (here) and an MLP (regression_nn.py, torch process)
    oof_c, (te_c, co_c) = crossfit_catboost(Xs["valid"], va.y.values, [Xs["test"], Xs["cold"]])
    for n, p in zip(sets, [oof_c, te_c, co_c]):
        preds[n]["CatBoost (stacked)"] = p
    log(f"[3.7] CatBoost OOF MSE {metrics.mse(va.y.values, oof_c):.4f} ({time.time() - t0:.0f}s)")
    has_nn = (ART / "reg_nn_valid.parquet").exists()
    if has_nn:
        for n in sets:
            preds[n]["MLP (neural net)"] = pd.read_parquet(ART / f"reg_nn_{n}.parquet")["MLP (neural net)"].values
        log(f"[3.7] MLP OOF MSE {metrics.mse(va.y.values, preds['valid']['MLP (neural net)']):.4f}")

    # ---- 3.6 ensemble: NON-NEGATIVE blend fit on valid OOF predictions (no offsetting weights)
    base = ["Bias model", "Latent factor (MF)", "Factorization machine", "LightGBM (stacked)", "Hurdle (LightGBM)",
            "CatBoost (stacked)"] + (["MLP (neural net)"] if has_nn else [])
    A = np.column_stack([preds["valid"][base].values, np.ones(len(va))])
    w, _ = nnls(A, va.y.values)
    for n in sets:
        preds[n]["Ensemble"] = np.column_stack([preds[n][base].values, np.ones(len(preds[n]))]) @ w
    tuning = {"blend_weights": dict(zip(base, w[:-1].tolist())), "blend_intercept": float(w[-1]),
              "fit_gap": FIT_GAP}
    log(f"[3.6] blend weights {tuning['blend_weights']}")
    log(f"[3.6] fit gaps {FIT_GAP}")

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
