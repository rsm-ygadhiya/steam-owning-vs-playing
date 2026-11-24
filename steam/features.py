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
