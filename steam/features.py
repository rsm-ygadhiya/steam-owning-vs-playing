"""Shared feature builders (item metadata, user/item stats, user x genre affinity).

Leakage rule: anything derived from playtime uses TRAIN rows only.
Ownership (which games are in a library) is a legitimate input and may use the
full library, because the task is "given a library, predict play".
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import ART
from . import data, splits

N_TAGS = 60


@lru_cache(None)
def item_meta() -> pd.DataFrame:
    """One row per item index with cleaned metadata (NaN where unknown)."""
    _, games, _, _ = data.load()
    items = pd.read_parquet(ART / "item_index.parquet")
    m = items.merge(games, on="item_id", how="left")
    m["has_meta"] = m.title.notna()
    for col in ["genres", "tags", "specs"]:
        m[col] = m[col].apply(lambda v: list(v) if isinstance(v, (list, np.ndarray)) else [])
    m["n_specs"] = m.specs.apply(len).astype(float)
    m["multiplayer"] = m.specs.apply(lambda s: any("Multi" in x for x in s)).astype(float)
    m.loc[~m.has_meta, ["n_specs", "multiplayer"]] = np.nan
    m["free"] = m.free.astype("float")
    m["early_access"] = m.early_access.astype("float")
    return m


@lru_cache(None)
def vocab():
    m = item_meta()
    genres = sorted({g for gs in m.genres for g in gs})
    tag_counts = pd.Series([t for ts in m.tags for t in ts]).value_counts()
    tags = list(tag_counts.index[:N_TAGS])
    return genres, tags


@lru_cache(None)
def genre_matrix() -> sp.csr_matrix:
    """items x genres binary matrix."""
    m = item_meta()
    genres, _ = vocab()
    gi = {g: k for k, g in enumerate(genres)}
    rows, cols = [], []
    for i, gs in enumerate(m.genres):
        for g in gs:
            rows.append(i); cols.append(gi[g])
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(m), len(genres)))


@lru_cache(None)
def tag_matrix() -> sp.csr_matrix:
    m = item_meta()
    _, tags = vocab()
    ti = {t: k for k, t in enumerate(tags)}
    rows, cols = [], []
    for i, ts in enumerate(m.tags):
        for t in ts:
            if t in ti:
                rows.append(i); cols.append(ti[t])
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(m), len(tags)))


@lru_cache(None)
def ownership() -> sp.csr_matrix:
    """users x items binary ownership over the FULL library (allowed input)."""
    st = splits.stats()
    parts = [splits.load(n)[["u", "i"]] for n in ["train", "valid", "test", "cold"]]
    d = pd.concat(parts)
    return sp.csr_matrix((np.ones(len(d), np.float32), (d.u.values, d.i.values)),
                         shape=(st["n_users"], st["n_items"]))


@lru_cache(None)
def train_stats():
    """User and item statistics computed from TRAIN only."""
    tr = splits.load("train")
    st = splits.stats()
    nu, ni = st["n_users"], st["n_items"]

    def agg(key, n):
        g = tr.groupby(key)
        out = pd.DataFrame(index=np.arange(n))
        out["n_train"] = g.size()
        out["mean_y"] = g.y.mean()
        out["played_rate"] = g.played.mean()
        out["active_rate"] = g.active.mean()
        out["mean_y_played"] = tr[tr.played == 1].groupby(key).y.mean()
        out["n_train"] = out.n_train.fillna(0)
        return out

    ustats, istats = agg("u", nu), agg("i", ni)
    lib = np.asarray(ownership().sum(1)).ravel()
    ustats["library_size"] = lib

    # user x genre affinity from train playtime, and genre share from ownership
    G = genre_matrix()
    Y = sp.csr_matrix((tr.y.values, (tr.u.values, tr.i.values)), shape=(nu, ni))
    C = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    ug_sum, ug_cnt = (Y @ G).toarray(), (C @ G).toarray()
    ug_mean = np.where(ug_cnt > 0, ug_sum / np.maximum(ug_cnt, 1), np.nan)
    ug_share = (ownership() @ G).toarray() / np.maximum(lib, 1)[:, None]
    return ustats, istats, ug_mean.astype(np.float32), ug_share.astype(np.float32)


def pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-free tabular features for (u, i) pairs; used by LightGBM / logistic models."""
    m = item_meta()
    ustats, istats, ug_mean, ug_share = train_stats()
    u, i = df.u.values, df.i.values
    X = pd.DataFrame(index=df.index)
    for c in ["n_train", "mean_y", "played_rate", "active_rate", "mean_y_played", "library_size"]:
        X[f"user_{c}"] = ustats[c].values[u]
    for c in ["n_train", "mean_y", "played_rate", "active_rate", "mean_y_played"]:
        X[f"item_{c}"] = istats[c].values[i]
    for c in ["price", "free", "release_year", "metascore", "early_access", "n_specs", "multiplayer"]:
        X[f"meta_{c}"] = m[c].values.astype(float)[i]
    # user x game genre affinity: average over the game's genres
    G = genre_matrix()
    gcount = np.asarray(G.sum(1)).ravel()
    Gi = G[i]                                     # rows = pairs
    with np.errstate(invalid="ignore"):
        aff = ug_mean[u]
        aff_masked = np.where(np.isnan(aff), 0, aff)
        has = (~np.isnan(aff)).astype(np.float32)
        num = np.asarray(Gi.multiply(aff_masked).sum(1)).ravel()
        den = np.asarray(Gi.multiply(has).sum(1)).ravel()
        X["ug_mean_y"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        X["ug_share"] = np.where(gcount[i] > 0,
                                 np.asarray(Gi.multiply(ug_share[u]).sum(1)).ravel() / np.maximum(gcount[i], 1),
                                 np.nan)
    return X


# =====================================================================================
# V3 features: variables the first versions did not use (developer, publisher, store
# sentiment, bundles, reviews, tags) and explicit user x game interactions.
# Same leakage rule: anything derived from playtime or reviews uses TRAIN pairs only.
# =====================================================================================
# columns that carry no game-identity signal -> allowed in the cold-start (metadata-only) model
META_PREFIXES = ("user_", "meta_", "ug_", "ut_", "udev_", "uc_", "x_")
EXTRA_VERSION = 1
SENTIMENT = {"Overwhelmingly Negative": 0, "Very Negative": 1, "Negative": 2, "Mostly Negative": 3, "Mixed": 4,
             "Mostly Positive": 5, "Positive": 6, "Very Positive": 7, "Overwhelmingly Positive": 8}


@lru_cache(None)
def content_matrix() -> sp.csr_matrix:
    """tf-idf over tags, genres, specs and developer: one 'document' per game (L2-normalized rows)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    m = item_meta()
    docs = [" ".join(["tag_" + t.replace(" ", "_") for t in tg] + ["genre_" + g.replace(" ", "_") for g in gs]
                     + ["spec_" + s.replace(" ", "_") for s in sc]
                     + (["dev_" + str(dv).replace(" ", "_")] if isinstance(dv, str) else []))
            for tg, gs, sc, dv in zip(m.tags, m.genres, m.specs, m.developer)]
    X = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True).fit_transform(docs)
    return normalize(X).tocsr()


def _first(v):
    if isinstance(v, (list, np.ndarray)):
        return v[0] if len(v) else None
    return v if isinstance(v, str) else None


@lru_cache(None)
def item_extra() -> pd.DataFrame:
    """Per-game metadata the first versions ignored + train-only developer/publisher/review encodings."""
    m = item_meta()
    _, _, reviews, bundles = data.load()
    tr = splits.load("train")
    ni = len(m)
    out = pd.DataFrame(index=np.arange(ni))
    sent = m.sentiment.astype("string")
    out["meta_sentiment"] = sent.map(SENTIMENT).astype(float).values
    n_small = sent.str.extract(r"^(\d+) user review")[0].astype(float)
    out["meta_few_store_reviews"] = n_small.values            # store page had < 10 reviews (else NaN)
    # bundles
    items = pd.read_parquet(ART / "item_index.parquet").reset_index().rename(columns={"index": "i"})
    b = bundles.merge(items, on="item_id")
    b["disc"] = pd.to_numeric(b.discount.astype(str).str.rstrip("%"), errors="coerce")
    out["meta_n_bundles"] = b.groupby("i").size().reindex(out.index).fillna(0).values
    out["meta_bundle_discount"] = b.groupby("i").disc.max().reindex(out.index).values
    # developer / publisher target encoding (smoothed, train playtime only), LEAVE-GAME-OUT:
    # a game's own playtime is subtracted from its developer's totals. Otherwise a one-game developer's
    # encoding equals the game's own mean, and the cold-start model (trained on warm games) would learn
    # to trust a signal that can never contain the game itself for never-seen games.
    gmean, k = tr.y.mean(), 20.0
    own = pd.DataFrame({"i": tr.i.values, "y": tr.y.values, "p": tr.played.values}).groupby("i").agg(
        s=("y", "sum"), n=("y", "size"), ps=("p", "sum")).reindex(out.index).fillna(0)
    for col in ["developer", "publisher"]:
        key = m[col].map(_first)
        g = own.groupby(key.values).sum()               # per developer: sum y, count, sum played
        tot = g.reindex(key.values).set_axis(out.index)
        s_, n_, ps_ = tot.s - own.s, tot.n - own.n, tot.ps - own.ps
        short = col[:3]
        out[f"meta_{short}_mean_y"] = np.where(key.notna(), (s_ + k * gmean) / (n_ + k), np.nan)
        out[f"meta_{short}_played_rate"] = np.where(key.notna() & (n_ > 0), ps_ / np.maximum(n_, 1), np.nan)
        out[f"meta_{short}_n_games"] = key.map(key.value_counts()).astype(float).values
    # reviews written on TRAIN pairs only (a review on a held-out pair would reveal that it was played)
    rv = _reviews_train()
    out["item_n_reviews"] = rv.groupby("i").size().reindex(out.index).fillna(0).values
    out["item_recommend_rate"] = rv.groupby("i").recommend.mean().reindex(out.index).values
    return out


@lru_cache(None)
def _reviews_train() -> pd.DataFrame:
    _, _, reviews, _ = data.load()
    users = pd.read_parquet(ART / "user_index.parquet").reset_index().rename(columns={"index": "u"})
    items = pd.read_parquet(ART / "item_index.parquet").reset_index().rename(columns={"index": "i"})
    rv = reviews.merge(users, on="user_id").merge(items, on="item_id")[["u", "i", "recommend", "helpful_votes"]]
    tr = splits.load("train")[["u", "i"]]
    return rv.merge(tr, on=["u", "i"])


@lru_cache(None)
def user_extra() -> pd.DataFrame:
    m = item_meta()
    st = splits.stats()
    O = ownership().tocsr()
    lib = np.maximum(np.asarray(O.sum(1)).ravel(), 1)
    out = pd.DataFrame(index=np.arange(st["n_users"]))
    out["user_free_share"] = (O @ m.free.fillna(0).values) / lib
    price = m.price.values.astype(float)
    has = ~np.isnan(price)
    out["user_mean_price"] = (O @ np.where(has, price, 0)) / np.maximum(O @ has.astype(float), 1)
    out["user_meta_share"] = (O @ m.has_meta.values.astype(float)) / lib
    rv = _reviews_train()
    out["user_n_reviews"] = rv.groupby("u").size().reindex(out.index).fillna(0).values
    out["user_recommend_rate"] = rv.groupby("u").recommend.mean().reindex(out.index).values
    return out


def _affinity(df, M, name):
    """User x attribute affinity: user's mean train log-playtime over games sharing the game's attributes."""
    tr = splits.load("train")
    st = splits.stats()
    nu, ni = st["n_users"], st["n_items"]
    Y = sp.csr_matrix((tr.y.values, (tr.u.values, tr.i.values)), shape=(nu, ni))
    C = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    S, N = (Y @ M).tocsr(), (C @ M).tocsr()        # users x attributes (sparse)
    Mi = M[df.i.values]
    num = np.asarray(S[df.u.values].multiply(Mi).sum(1)).ravel()
    den = np.asarray(N[df.u.values].multiply(Mi).sum(1)).ravel()
    return pd.Series(np.where(den > 0, num / np.maximum(den, 1e-9), np.nan), index=df.index, name=name), den


def _knn(df, V, k=20):
    """Similarity-weighted average of the user's own train log-playtime over the k most similar
    games they own (item-based CF with learned or content similarity). Returns knn_y, knn_played, max_sim."""
    tr = splits.load("train").sort_values("u")
    ui = np.searchsorted(tr.u.values, np.arange(splits.stats()["n_users"] + 1))
    ti, ty, tp = tr.i.values, tr.y.values, tr.played.values.astype(float)
    out = np.full((len(df), 3), np.nan)
    order = np.argsort(df.u.values, kind="stable")
    du, di = df.u.values[order], df.i.values[order]
    bounds = np.flatnonzero(np.diff(du)) + 1
    sparse_V = sp.issparse(V)
    for grp in np.split(np.arange(len(du)), bounds):
        u = du[grp[0]]
        a, b = ui[u], ui[u + 1]
        if b - a == 0:
            continue
        J, T = ti[a:b], di[grp]
        Sm = (V[T] @ V[J].T).toarray() if sparse_V else V[T] @ V[J].T
        Sm = np.maximum(Sm, 0)
        if Sm.shape[1] > k:
            thr = -np.partition(-Sm, k - 1, axis=1)[:, k - 1:k]
            Sm = np.where(Sm >= thr, Sm, 0)
        w = Sm.sum(1)
        ok = w > 1e-9
        res = np.full((len(T), 3), np.nan)
        res[ok, 0] = (Sm[ok] @ ty[a:b]) / w[ok]
        res[ok, 1] = (Sm[ok] @ tp[a:b]) / w[ok]
        res[:, 2] = Sm.max(1)
        out[order[grp]] = res
    return out


def extra_features(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """All V3 features for one split, cached to artifacts/feat_extra_{name}.parquet."""
    path = ART / f"feat_extra_{name}_v{EXTRA_VERSION}.parquet"
    if path.exists():
        X = pd.read_parquet(path)
        if len(X) == len(df):
            return X.set_index(df.index)
    u, i = df.u.values, df.i.values
    ie, ue = item_extra(), user_extra()
    X = pd.DataFrame(index=df.index)
    for c in ue.columns:
        X[c] = ue[c].values[u]
    for c in ie.columns:
        X[c] = ie[c].values[i]
    m = item_meta()
    X["x_price_vs_user"] = m.price.values.astype(float)[i] / (X.user_mean_price + 1.0)
    X["ut_mean_y"], X["ut_n"] = _affinity(df, tag_matrix(), "ut_mean_y")
    # same developer: did this user play the developer's other games?
    dev = m.developer.map(_first)
    codes, _ = pd.factorize(dev)
    D = sp.csr_matrix((np.ones((codes >= 0).sum(), np.float32), (np.flatnonzero(codes >= 0), codes[codes >= 0])),
                      shape=(len(m), max(codes.max() + 1, 1)))
    X["udev_mean_y"], X["udev_n"] = _affinity(df, D, "udev_mean_y")
    # item-based kNN over the user's own games: content similarity (works for never-seen games) ...
    knn_c = _knn(df, content_matrix())
    X["uc_knn_y"], X["uc_knn_played"], X["uc_max_sim"] = knn_c.T
    # ... and learned item2vec similarity (warm games only: game identity -> excluded from the cold model)
    V = np.load(ART / "item2vec_vectors.npy")
    knn_v = _knn(df, V)
    X["cf_knn_y"], X["cf_knn_played"], X["cf_max_sim"] = knn_v.T
    cold = np.isin(i, np.load(ART / "cold_items.npy"))
    X.loc[cold, ["cf_knn_y", "cf_knn_played", "cf_max_sim"]] = np.nan
    X.reset_index(drop=True).astype(np.float32).to_parquet(path)
    return X


def stack_features(name: str, d: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Leak-free pair features + V3 extra features + stage-A model outputs."""
    X = pair_features(d)
    X = pd.concat([X, extra_features(name, d)], axis=1)
    X["bias_pred"] = base["Bias model"].values
    X["mf_pred"] = base["Latent factor (MF)"].values
    X["mf_dot"] = base["_mf_dot"].values
    X["fm_pred"] = base["Factorization machine"].values
    # implicit-feedback affinities from the ranking stage (both fit on TRAIN played pairs only)
    cold = np.isin(d.i.values, np.load(ART / "cold_items.npy"))
    for col, (uf, vf) in {"als_dot": ("als_user_factors", "als_item_factors"),
                          "i2v_sim": ("item2vec_user", "item2vec_vectors")}.items():
        U, V = np.load(ART / f"{uf}.npy"), np.load(ART / f"{vf}.npy")
        v = np.einsum("nk,nk->n", U[d.u.values], V[d.i.values]).astype(np.float64)
        v[cold] = np.nan                    # never-seen games have no trained factors
        X[col] = v
    return X
