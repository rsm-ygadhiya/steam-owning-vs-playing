"""Task 5: top-N recommendation of games a user will actually PLAY.

Training signal: played (playtime > 0) pairs from TRAIN only.
Protocols (built in splits.py, all from held-out rows):
  valid / test : 1 played held-out game vs 99 games the user does not own
  hard         : 1 played held-out game vs 9 owned-but-unplayed held-out games
  cold         : 1 played never-seen game vs 99 never-seen games not owned

Models
  5.0 random, most-popular, Jaccard item-kNN                    (course baselines)
  5.1 BPR-MF                                                    (Rendle 2009)
  5.2 implicit ALS, binary vs playtime-weighted confidence     (Hu, Koren & Volinsky 2008)
  5.3 item2vec on libraries ordered by playtime                 (Barkan & Koenigstein 2016)
  5.4 content (tf-idf over tags/genres/specs/developer), content->ALS factor mapping
      for cold start, and a warm hybrid of ALS + content
"""
from __future__ import annotations

import json
import time

import numpy as np
import scipy.sparse as sp
from gensim.models import Word2Vec
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

from .config import ART, SEED, SEEDS
from . import features, metrics, splits

RES = ART / "results"
RES.mkdir(exist_ok=True)

ALS_REGS = [0.05, 1.0, 5.0]
BPR_GRID = [(64, 0.01, 0.01, 100), (64, 0.005, 0.01, 200), (128, 0.01, 0.001, 200), (128, 0.005, 0.01, 200)]


# ----------------------------------------------------------------- data
def train_matrices():
    tr = splits.load("train")
    st = splits.stats()
    p = tr[tr.played == 1]
    shape = (st["n_users"], st["n_items"])
    R = sp.csr_matrix((np.ones(len(p), np.float32), (p.u.values, p.i.values)), shape=shape)
    W = sp.csr_matrix((np.log1p(p.playtime.values).astype(np.float32), (p.u.values, p.i.values)), shape=shape)
    return R, W


# ------------------------------------------------------------- scorers
# Every model is a function score(users, cands) -> (n_users, n_cands) array.

def factor_scorer(U, V):
    return lambda users, cands: np.einsum("uk,unk->un", U[users], V[cands])


def popularity_scorer(pop):
    return lambda users, cands: pop[cands].astype(np.float64)


def random_scorer(seed=SEED):
    rng = np.random.default_rng(seed)
    return lambda users, cands: rng.random(cands.shape)


def jaccard_scorer(R):
    """score(u, j) = sum over i in I_u of Jaccard(i, j); item-item matrix is sparse."""
    Rb = R.astype(bool).astype(np.float32).tocsc()
    C = (Rb.T @ Rb).tocsr()
    n = np.asarray(Rb.sum(0)).ravel()
    C = C.tocoo()
    J = sp.csr_matrix((C.data / (n[C.row] + n[C.col] - C.data), (C.row, C.col)), shape=C.shape)
    J.setdiag(0)
    J.eliminate_zeros()
    Ru = R.tocsr()

    def score(users, cands, bs=2048):
        out = np.empty(cands.shape)
        for s in range(0, len(users), bs):
            S = (Ru[users[s:s + bs]] @ J).toarray()
            out[s:s + bs] = np.take_along_axis(S, cands[s:s + bs], 1)
        return out
    return score


def fit_als(M, factors=64, reg=0.05, alpha=20.0, iters=15, seed=SEED):
    m = AlternatingLeastSquares(factors=factors, regularization=reg, alpha=alpha,
                                iterations=iters, random_state=seed)
    m.fit(M, show_progress=False)
    return m.user_factors, m.item_factors


def fit_bpr(R, factors=64, lr=0.05, reg=0.01, iters=100, seed=SEED):
    m = BayesianPersonalizedRanking(factors=factors, learning_rate=lr, regularization=reg,
                                    iterations=iters, random_state=seed)
    m.fit(R, show_progress=False)
    return m.user_factors, m.item_factors


def fit_item2vec(W, dim=64, window=10, epochs=10, seed=SEED):
    """Each user's played games, sorted by playtime (desc), form a 'sentence'."""
    W = W.tocsr()
    sents = []
    for u in range(W.shape[0]):
        a, b = W.indptr[u], W.indptr[u + 1]
        if b - a >= 2:
            order = np.argsort(-W.data[a:b])
            sents.append([str(x) for x in W.indices[a:b][order]])
    w2v = Word2Vec(sents, vector_size=dim, window=window, min_count=1, sg=1, negative=10,
                   epochs=epochs, workers=8, seed=seed)
    V = np.zeros((W.shape[1], dim), np.float32)
    for tok in w2v.wv.index_to_key:
        V[int(tok)] = w2v.wv[tok]
    V = normalize(V)
    U = normalize(W @ V)                    # playtime-weighted average of game vectors
    return U, V


def content_matrix():
    """tf-idf over tags, genres, specs and developer: one 'document' per game."""
    m = features.item_meta()
    docs = [" ".join(["tag_" + t.replace(" ", "_") for t in tg] + ["genre_" + g.replace(" ", "_") for g in gs]
                     + ["spec_" + s.replace(" ", "_") for s in sc]
                     + (["dev_" + str(dv).replace(" ", "_")] if isinstance(dv, str) else []))
            for tg, gs, sc, dv in zip(m.tags, m.genres, m.specs, m.developer)]
    X = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit_transform(docs)
    return normalize(X).tocsr()


def content_scorer(W, X):
    """User profile = playtime-weighted sum of game tf-idf vectors; score = cosine."""
    P = normalize(W @ X)

    def score(users, cands):
        Pu = P[users]
        return np.asarray([(X[c] @ Pu[r].T).toarray().ravel() for r, c in enumerate(cands)])
    return score


def mapped_factor_scorer(U, V, X, warm_items, alpha=1.0):
    """Cold start: learn a ridge map from content to ALS item factors on warm games,
    then give never-seen games a factor vector predicted from their content."""
    ridge = Ridge(alpha=alpha).fit(X[warm_items], V[warm_items])
    V_hat = V.copy()
    V_hat[:] = ridge.predict(X)
    return factor_scorer(U, V_hat), V_hat


def zscore_rows(S):
    """Per-user standardization: keeps score gaps (rank-normalizing throws them away)."""
    S = np.asarray(S, np.float64)
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def hybrid_scorer(s1, s2, w, norm=zscore_rows):
    return lambda users, cands: (1 - w) * norm(s1(users, cands)) + w * norm(s2(users, cands))


# Candidate scores that the learning-to-rank stage (ranking_ltr.py, LightGBM) combines.
LTR_MODELS = ["Most popular", "Jaccard item-kNN", "ALS (binary)", "ALS (playtime-weighted)",
              "BPR-MF", "item2vec", "Content (tf-idf)"]


def dump_candidates(scorers, R, path, bs=5000):
    """Save every base model's score for every candidate: (n_users, n_cands, n_models)."""
    users, cands = R["users"], np.concatenate([R["pos"][:, None], R["negs"]], 1)
    F = np.zeros(cands.shape + (len(LTR_MODELS),), np.float32)
    for k, n in enumerate(LTR_MODELS):
        for s in range(0, len(users), bs):
            F[s:s + bs, :, k] = scorers[n](users[s:s + bs], cands[s:s + bs])
    np.savez(path, users=users, cands=cands, F=F, names=np.array(LTR_MODELS))


# ----------------------------------------------------------- evaluation
def evaluate(scorer, R, ks=None):
    """ks defaults to (1, 10), or (1, 3) for the 10-candidate hard protocol."""
    if ks is None:
        ks = (1, 3) if R["negs"].shape[1] < 20 else (1, 10)
    users, pos, negs = R["users"], R["pos"], R["negs"]
    cands = np.concatenate([pos[:, None], negs], 1)
    S = scorer(users, cands)
    return metrics.ranking_report(S[:, 0], S[:, 1:], ks=ks)


# ------------------------------------------------------------------ main
def run(log=print, quick=False):
    t0 = time.time()
    st = splits.stats()
    ni = st["n_items"]
    R, W = train_matrices()
    pop = np.asarray(R.sum(0)).ravel()
    P = {n: splits.load_rank(n) for n in ["valid", "test", "hard", "cold"]}
    if quick:
        P = {n: {k: v[:3000] for k, v in d.items()} for n, d in P.items()}
    cold_items = np.load(ART / "cold_items.npy")
    warm = np.setdiff1d(np.arange(ni), cold_items)
    results = {n: {} for n in P}
    tuning = {}
    scorers = {}

    def ev(name, scorer, protos=("test", "hard")):
        for p in protos:
            results[p][name] = evaluate(scorer, P[p])
        log(f"[5] {name}: " + "  ".join(f"{p} AUC={results[p][name]['AUC']:.4f} NDCG@10/3={results[p][name].get('NDCG@10', results[p][name].get('NDCG@3')):.4f}"
                                        for p in protos) + f" ({time.time() - t0:.0f}s)")

    # ---- 5.0 baselines
    scorers["Random"] = random_scorer()
    scorers["Most popular"] = popularity_scorer(pop)
    scorers["Jaccard item-kNN"] = jaccard_scorer(R)
    for n in ["Random", "Most popular", "Jaccard item-kNN"]:
        ev(n, scorers[n])

    # ---- 5.2 ALS: tune alpha on VALID, binary vs playtime-weighted
    tuning["als"] = []
    best = {}
    for kind, M in [("binary", R), ("playtime", W)]:
        # V1 found f=64 >> f=128 at reg=0.05: a sign of under-regularization, so reg is tuned too.
        grid = [(10, 32, 0.05)] if quick else [(a, f, r) for a in [20, 60] for f in [64, 128] for r in ALS_REGS]
        for alpha, f, reg in grid:
            U, V = fit_als(M, factors=f, reg=reg, alpha=alpha)
            v = evaluate(factor_scorer(U, V), P["valid"])
            tuning["als"].append({"confidence": kind, "alpha": alpha, "factors": f, "reg": reg, **v})
            log(f"[5.2] ALS {kind} alpha={alpha} f={f} reg={reg}: valid NDCG@10={v['NDCG@10']:.4f} ({time.time() - t0:.0f}s)")
            if v["NDCG@10"] > best.get(kind, (-1,))[0]:
                best[kind] = (v["NDCG@10"], alpha, f, reg, U, V)
    for kind, label in [("binary", "ALS (binary)"), ("playtime", "ALS (playtime-weighted)")]:
        _, alpha, f, reg, U, V = best[kind]
        scorers[label] = factor_scorer(U, V)
        ev(label, scorers[label])
    als_U, als_V = best["playtime"][4], best["playtime"][5]
    np.save(ART / "als_item_factors.npy", als_V)
    np.save(ART / "als_user_factors.npy", als_U)       # reused as stacking features (regression_stack)

    # ---- 5.1 BPR
    tuning["bpr"] = []
    bestb = (-1,)
    # V1: lr 0.01 beat 0.05 -> explore smaller steps, more epochs and the L2 strength.
    for f, lr, reg, it in ([(32, 0.05, 0.01, 30)] if quick else BPR_GRID):
        U, V = fit_bpr(R, factors=f, lr=lr, reg=reg, iters=it)
        v = evaluate(factor_scorer(U, V), P["valid"])
        tuning["bpr"].append({"factors": f, "lr": lr, "reg": reg, "iters": it, **v})
        log(f"[5.1] BPR f={f} lr={lr} reg={reg} iters={it}: valid NDCG@10={v['NDCG@10']:.4f} ({time.time() - t0:.0f}s)")
        if v["NDCG@10"] > bestb[0]:
            bestb = (v["NDCG@10"], f, lr, reg, it, U, V)
    scorers["BPR-MF"] = factor_scorer(bestb[5], bestb[6])
    ev("BPR-MF", scorers["BPR-MF"])

    # ---- 5.3 item2vec
    U, V = fit_item2vec(W, epochs=3 if quick else 10)
    scorers["item2vec"] = factor_scorer(U, V)
    ev("item2vec", scorers["item2vec"])
    np.save(ART / "item2vec_vectors.npy", V)
    np.save(ART / "item2vec_user.npy", U)

    # ---- 5.4 content, hybrid, cold-start mapping
    X = content_matrix()
    scorers["Content (tf-idf)"] = content_scorer(W, X)
    ev("Content (tf-idf)", scorers["Content (tf-idf)"])
    tuning["hybrid"] = []
    bh = (-1, 0)
    for w in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        v = evaluate(hybrid_scorer(scorers["ALS (playtime-weighted)"], scorers["Content (tf-idf)"], w), P["valid"])
        tuning["hybrid"].append({"w_content": w, **v})
        log(f"[5.4] hybrid w_content={w}: valid NDCG@10={v['NDCG@10']:.4f}")
        if v["NDCG@10"] > bh[0]:
            bh = (v["NDCG@10"], w)
    scorers["Hybrid (ALS + content)"] = hybrid_scorer(scorers["ALS (playtime-weighted)"], scorers["Content (tf-idf)"], bh[1])
    ev("Hybrid (ALS + content)", scorers["Hybrid (ALS + content)"])

    # cold start: only content-aware models can score never-seen games
    has_meta = features.item_meta().has_meta.values
    mapped, _ = mapped_factor_scorer(als_U, als_V, X, warm[has_meta[warm]])
    pop_model = Ridge(alpha=1.0).fit(X[warm], np.log1p(pop[warm]))
    pred_pop = pop_model.predict(X)
    cold_scorers = {
        "Random": scorers["Random"],
        "Most popular": scorers["Most popular"],       # all cold games have 0 plays: ties
        "Predicted popularity (content)": popularity_scorer(pred_pop),
        "Content (tf-idf)": scorers["Content (tf-idf)"],
        "ALS factors mapped from content": mapped,
        "Hybrid (mapped ALS + content)": hybrid_scorer(mapped, scorers["Content (tf-idf)"], 0.5),
    }
    for n, s in cold_scorers.items():
        results["cold"][n] = evaluate(s, P["cold"])
        log(f"[5.4] cold {n}: AUC={results['cold'][n]['AUC']:.4f} HR@10={results['cold'][n]['HR@10']:.4f}")

    # ---- candidate scores for the learning-to-rank stage (valid trains it, test/hard evaluate it)
    for p in ["valid", "test", "hard"]:
        dump_candidates(scorers, P[p], ART / f"rank_feats_{p}.npz")
    log(f"[5] candidate features saved ({time.time() - t0:.0f}s)")

    # ---- error bars over seeds for the factor models (test protocol)
    seeds = {}
    for label, fit in [("ALS (playtime-weighted)", lambda s: fit_als(W, best["playtime"][2], reg=best["playtime"][3], alpha=best["playtime"][1], seed=s)),
                       ("ALS (binary)", lambda s: fit_als(R, best["binary"][2], reg=best["binary"][3], alpha=best["binary"][1], seed=s)),
                       ("BPR-MF", lambda s: fit_bpr(R, bestb[1], lr=bestb[2], reg=bestb[3], iters=bestb[4], seed=s))]:
        runs = [evaluate(factor_scorer(*fit(s)), P["test"]) for s in (SEEDS[:1] if quick else SEEDS)]
        seeds[label] = {k: [r[k] for r in runs] for k in runs[0]}
    log(f"[5] seeds done ({time.time() - t0:.0f}s)")

    # ---- beyond accuracy: top-10 over the full warm catalog for 3000 test users
    rng = np.random.default_rng(SEED)
    users = rng.choice(P["test"]["users"], size=min(3000, len(P["test"]["users"])), replace=False)
    owned = features.ownership().tocsr()
    allc = np.tile(warm, (len(users), 1))
    beyond, tops = {}, {}
    for n in ["Most popular", "Jaccard item-kNN", "BPR-MF", "ALS (binary)", "ALS (playtime-weighted)",
              "item2vec", "Content (tf-idf)", "Hybrid (ALS + content)"]:
        S = np.vstack([scorers[n](users[s:s + 250], allc[s:s + 250]) for s in range(0, len(users), 250)])
        S = S.astype(np.float64)
        S[owned[users][:, warm].nonzero()] = -np.inf
        top = warm[np.argsort(-S, 1)[:, :10]]
        tops[n] = top
        beyond[n] = metrics.beyond_accuracy(top, pop, ni)
    np.savez(ART / "rank_top10.npz", users=users, **{k.replace(" ", "_"): v for k, v in tops.items()})
    log(f"[5] beyond-accuracy done ({time.time() - t0:.0f}s)")

    with open(RES / "ranking.json", "w") as f:
        json.dump({"results": results, "tuning": tuning, "seeds": seeds, "beyond": beyond,
                   "best": {"als_playtime": dict(zip(["alpha", "factors", "reg"], best["playtime"][1:4])),
                            "als_binary": dict(zip(["alpha", "factors", "reg"], best["binary"][1:4])),
                            "bpr": dict(zip(["factors", "lr", "reg", "iters"], bestb[1:5])),
                            "hybrid_w": bh[1]}}, f, indent=1)
    log(f"[5] done in {time.time() - t0:.0f}s")
    return results


if __name__ == "__main__":
    import sys
    run(quick="--quick" in sys.argv)
