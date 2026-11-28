"""Task 5 (stage B, LightGBM): learning-to-rank over every base recommender + Task 4 on the hard protocol.

5.5 LambdaRank reranker
    Each candidate (user, game) gets the scores of all base models from ranking.py
    (popularity, Jaccard, both ALS variants, BPR, item2vec, content), their per-user
    z-scores, and game metadata. A LightGBM LambdaRank model (Burges 2010) learns how to
    combine them. It is trained on the VALID protocol (held-out valid positives vs 99
    not-owned games) and evaluated on TEST and HARD, which hold different held-out games.
    The base models were fit on TRAIN only, so no candidate score has seen its label.

5.6 Task 4 classifier as a ranker on the hard protocol
    The hard protocol (1 played vs 9 owned-but-unplayed games) is exactly the "will they
    play it?" question asked per user. All of its candidates are TEST pairs, so the
    LightGBM played-classifier's test predictions (cross-fit on VALID, regression_stack.py)
    can rank them directly.

Never import torch here (macOS OpenMP clash).
"""
from __future__ import annotations

import json
import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import ART, SEED
from . import features, metrics, splits

RES = ART / "results"

LTR = dict(objective="lambdarank", metric="None", learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambdarank_truncation_level=20, verbose=-1, seed=SEED, num_threads=8)


def candidate_features(z):
    """(n_users * n_cands, n_features) matrix + names, from a rank_feats_*.npz file."""
    F, cands, users = z["F"].astype(np.float64), z["cands"], z["users"]
    names = [str(n) for n in z["names"]]
    n, c, k = F.shape
    Z = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-9)    # per-user z-scores
    m = features.item_meta()
    ustats = features.train_stats()[0]
    meta = np.stack([m.price.values, m.release_year.values, m.metascore.values,
                     m.free.values, m.early_access.values, m.n_specs.values], 1).astype(np.float64)
    cols = [F.reshape(-1, k), Z.reshape(-1, k), meta[cands.ravel()],
            np.repeat(ustats.n_train.values[users], c)[:, None].astype(np.float64)]
    names = names + [f"z: {x}" for x in names] + ["meta_price", "meta_release_year", "meta_metascore",
                                                  "meta_free", "meta_early_access", "meta_n_specs",
                                                  "user_n_train"]
    return pd.DataFrame(np.hstack(cols), columns=names), (n, c)


def safe(df):
    """LightGBM rejects JSON-special characters in feature names; train on sanitized copies."""
    return df.set_axis([re.sub(r"[^0-9A-Za-z_]", "_", c) for c in df.columns], axis=1)


def user_ndcg(S, k=10):
    """Per-user NDCG@k for a (users, 1 + negatives) score matrix with the positive in column 0
    (pessimistic ties, same convention as metrics.ranking_report)."""
    rank = (S[:, 1:] >= S[:, :1]).sum(1)
    return np.where(rank < k, 1.0 / np.log2(rank + 2), 0.0)


def paired_bootstrap(a, b, n_boot=1000, seed=SEED):
    """95% CI of mean(a - b), resampling users with replacement."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975))}


def run(log=print):
    t0 = time.time()
    Z = {p: np.load(ART / f"rank_feats_{p}.npz") for p in ["valid", "test", "hard"]}
    X = {p: candidate_features(z) for p, z in Z.items()}
    results = {"test": {}, "hard": {}}

    # ---- 5.5 LambdaRank: train on valid users (10% of them held out for early stopping)
    Xv, (n, c) = X["valid"]
    y = np.zeros((n, c)); y[:, 0] = 1
    rng = np.random.default_rng(SEED)
    hold = rng.random(n) < 0.1
    rows = lambda mask: np.repeat(mask, c)
    dtr = lgb.Dataset(safe(Xv[rows(~hold)]), y[~hold].ravel(), group=np.full((~hold).sum(), c))
    dva = lgb.Dataset(safe(Xv[rows(hold)]), y[hold].ravel(), group=np.full(hold.sum(), c))
    # LightGBM's built-in NDCG breaks ties optimistically, so an untrained model (all scores tied)
    # looks perfect and early-stops at round 1. Use our pessimistic-tie NDCG@10 instead.
    def feval(pred, ds):
        return "ndcg10_pessimistic", float(user_ndcg(pred.reshape(-1, c)).mean()), True
    bst = lgb.train(LTR, dtr, 2000, valid_sets=[dva], feval=feval,
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    log(f"[5.5] LambdaRank: {bst.best_iteration} rounds, held-out valid NDCG@10 "
        f"{bst.best_score['valid_0']['ndcg10_pessimistic']:.4f} ({time.time() - t0:.0f}s)")
    for p in ["test", "hard"]:
        Xp, (n_p, c_p) = X[p]
        S = bst.predict(safe(Xp), num_iteration=bst.best_iteration).reshape(n_p, c_p)
        results[p]["LTR reranker (LightGBM)"] = metrics.ranking_report(
            S[:, 0], S[:, 1:], ks=(1, 3) if c_p < 20 else (1, 10))
        log(f"[5.5] {p}: {results[p]['LTR reranker (LightGBM)']}")
    imp = pd.Series(bst.feature_importance("gain"), index=Xv.columns).sort_values(ascending=False)

    # paired bootstrap over test users: is the reranker's gain over each base model real?
    Xt, (n_t, c_t) = X["test"]
    S_ltr = bst.predict(safe(Xt), num_iteration=bst.best_iteration).reshape(n_t, c_t)
    base_scores = {str(nm): Z["test"]["F"][:, :, k] for k, nm in enumerate(Z["test"]["names"])}
    paired = {nm: paired_bootstrap(user_ndcg(S_ltr), user_ndcg(S)) for nm, S in base_scores.items()}
    log("[5.5] paired bootstrap (LTR minus base, NDCG@10): "
        + ", ".join(f"{k}: {v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for k, v in paired.items()))

    # ---- 5.6 the Task 4 played-classifier on the hard protocol (all candidates are test pairs)
    te = splits.load("test")
    prob = np.load(ART / "played_probs_lgb.npz")["test"]
    lookup = pd.Series(prob, index=pd.MultiIndex.from_arrays([te.u.values, te.i.values]))
    zh = Z["hard"]
    users, cands = zh["users"], zh["cands"]
    S = lookup.reindex(pd.MultiIndex.from_arrays([np.repeat(users, cands.shape[1]), cands.ravel()])).values
    assert not np.isnan(S).any(), "every hard candidate should be a test pair"
    S = S.reshape(cands.shape)
    results["hard"]["Played classifier (Task 4 LightGBM)"] = metrics.ranking_report(S[:, 0], S[:, 1:], ks=(1, 3))
    log(f"[5.6] hard, played classifier: {results['hard']['Played classifier (Task 4 LightGBM)']}")

    with open(RES / "ranking_ltr.json", "w") as f:
        json.dump({"results": results, "importance": imp.to_dict(), "rounds": int(bst.best_iteration),
                   "paired_ndcg10_test": paired}, f, indent=1)
    log(f"[5.5] done in {time.time() - t0:.0f}s")
    return results


if __name__ == "__main__":
    run()
