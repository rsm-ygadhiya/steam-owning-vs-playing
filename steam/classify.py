"""Task 4: "Will they play it?" classification.

Targets
  played = playtime_forever > 0   (64/36 split)
  active = playtime_2weeks  > 0   (~3% positive: currently playing)

Models (all fit on VALID with 5-fold cross-fitting, features from TRAIN only)
  - item played-rate / user played-rate baselines (no fitting)
  - logistic regression on metadata + user features (course week 2)
  - logistic regression + collaborative-filtering scores (bias / MF / FM outputs)
  - LightGBM on the same stacked features (= stage 1 of the hurdle model)

Leakage: for `active`, no feature uses the row's own playtime. All CF scores
come from models fit on TRAIN rows, which never include the row being scored.
BER thresholds are chosen on VALID OOF predictions, then applied to test.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ART, SEED
from . import features, metrics, splits
from .regression_stack import LGB_CLF, crossfit_lgb, stack_features

RES = ART / "results"


def crossfit_logreg(X_va, y_va, X_others, folds=5, C=1.0):
    oof = np.zeros(len(X_va))
    others = [np.zeros(len(X)) for X in X_others]
    for a, b in KFold(folds, shuffle=True, random_state=SEED).split(X_va):
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(C=C, max_iter=300))
        m.fit(X_va.iloc[a], y_va[a])
        oof[b] = m.predict_proba(X_va.iloc[b])[:, 1]
        for o, X in zip(others, X_others):
            o += m.predict_proba(X)[:, 1] / folds
    return oof, others


def run(log=print):
    t0 = time.time()
    va, te = splits.load("valid"), splits.load("test")
    base = {n: pd.read_parquet(ART / f"reg_base_{n}.parquet") for n in ["valid", "test"]}
    Xva, Xte = stack_features("valid", va, base["valid"]), stack_features("test", te, base["test"])
    meta_cols = [c for c in Xva.columns if c.startswith(("user_", "meta_", "ug_"))]
    _, istats, _, _ = features.train_stats()
    ustats = features.train_stats()[0]

    results, probs = {}, {}
    for target in ["played", "active"]:
        yv, yt = va[target].values, te[target].values
        P = {}   # name -> (valid OOF, test)
        rate = f"{target}_rate"
        P["Item rate (baseline)"] = (istats[rate].fillna(yv.mean()).values[va.i.values],
                                     istats[rate].fillna(yv.mean()).values[te.i.values])
        P["User rate (baseline)"] = (ustats[rate].fillna(yv.mean()).values[va.u.values],
                                     ustats[rate].fillna(yv.mean()).values[te.u.values])
        oof, (pt,) = crossfit_logreg(Xva[meta_cols], yv, [Xte[meta_cols]])
        P["Logistic (features)"] = (oof, pt)
        oof, (pt,) = crossfit_logreg(Xva, yv, [Xte])
        P["Logistic (+ CF scores)"] = (oof, pt)
        log(f"[4] {target}: logistic done ({time.time() - t0:.0f}s)")
        if target == "played":
            z = np.load(ART / "played_probs_lgb.npz")   # reuse hurdle stage 1
            P["LightGBM (stacked)"] = (z["valid"], z["test"])
        else:
            oof, (pt,), _ = crossfit_lgb(Xva, yv, [Xte], LGB_CLF)
            P["LightGBM (stacked)"] = (oof, pt)
        log(f"[4] {target}: LightGBM done ({time.time() - t0:.0f}s)")

        results[target] = {}
        for name, (pv, pt) in P.items():
            thr = metrics.best_threshold_ber(yv, pv)        # chosen on valid only
            r = metrics.classification_report(yt, pt, thr)
            r["threshold"] = thr
            results[target][name] = r
        probs[target] = pd.DataFrame({k: v[1] for k, v in P.items()}).astype(np.float32)
        probs[target]["y"] = yt
        probs[target].to_parquet(ART / f"cls_probs_{target}.parquet")

    with open(RES / "classification.json", "w") as f:
        json.dump(results, f, indent=1)
    log(f"[4] done in {time.time() - t0:.0f}s")
    return results


if __name__ == "__main__":
    run()
