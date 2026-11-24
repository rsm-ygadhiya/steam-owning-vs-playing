"""Task 6: one evaluation module shared by every task."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- regression
def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def regression_report(y, p):
    return {"MSE": mse(y, p), "MAE": mae(y, p)}


# ------------------------------------------------------------ classification
def ber(y, p_hat, thr=0.5):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p_hat) >= thr
    fpr = np.mean(pred[~y]) if (~y).any() else 0.0
    fnr = np.mean(~pred[y]) if y.any() else 0.0
    return float(0.5 * (fpr + fnr))


def best_threshold_ber(y, p_hat):
    """BER depends on the threshold; choose it on VALIDATION data only."""
    grid = np.quantile(p_hat, np.linspace(0.02, 0.98, 49))
    bers = [ber(y, p_hat, t) for t in grid]
    return float(grid[int(np.argmin(bers))])


def classification_report(y, p_hat, thr=0.5):
    return {"AUC": float(roc_auc_score(y, p_hat)), "BER": ber(y, p_hat, thr),
            "Accuracy": float(np.mean((np.asarray(p_hat) >= thr) == np.asarray(y).astype(bool)))}


# ------------------------------------------------------------------- ranking
def ranking_report(pos_scores, neg_scores, ks=(1, 10)):
    """Leave-one-out style metrics.

    pos_scores: (n_users,) score of the held-out positive
    neg_scores: (n_users, n_neg) scores of that user's sampled negatives
    Ties are broken pessimistically (a tie counts as ranked above the positive)
    so a constant scorer cannot look good.
    """
    pos = np.asarray(pos_scores, dtype=np.float64)[:, None]
    neg = np.asarray(neg_scores, dtype=np.float64)
    rank = (neg >= pos).sum(1)                  # 0 = positive ranked first
    n_neg = neg.shape[1]
    auc = 1.0 - ((neg > pos).sum(1) + 0.5 * (neg == pos).sum(1)) / n_neg
    out = {"AUC": float(auc.mean())}
    for k in ks:
        hit = rank < k
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(rank + 2), 0.0).mean())
    return out


# --------------------------------------------------- beyond-accuracy metrics
def beyond_accuracy(topk: np.ndarray, item_pop: np.ndarray, n_items: int, head_frac=0.01):
    """Catalog coverage, head share and novelty for a matrix of top-K recommendations.

    topk:     (n_users, K) recommended item indices
    item_pop: (n_items,) training popularity (number of players)
    """
    flat = topk.ravel()
    order = np.argsort(-item_pop)
    head = np.zeros(n_items, bool)
    head[order[: max(1, int(head_frac * n_items))]] = True
    p = (item_pop + 1) / (item_pop.sum() + n_items)
    return {"coverage": float(len(np.unique(flat)) / n_items),
            "head_share": float(head[flat].mean()),
            "novelty": float(np.mean(-np.log2(p[flat])))}
