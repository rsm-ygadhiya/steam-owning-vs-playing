"""Task 1b: one shared, leak-free split used by EVERY model and task.

Design
------
1. Cold-start games: 10% of games that have metadata are removed from every
   training set. All their pairs form `cold` (the cold-start test set).
2. The remaining warm pairs are split at random 80/10/10 into train/valid/test.
   Every model (regression, classification, ranking) is fit on `train` only.
   `valid` is for tuning and for training stacked models; `test` is touched once.
3. Ranking protocols are built from the SAME held-out rows, so no ranking
   positive or hard negative is ever seen in training:
   - standard: per user, 1 played game from `test` + 99 warm games the user does
     NOT own (uniformly sampled). Measures "will they own and play it".
   - hard:     per user, 1 played game from `test` + 9 owned-but-UNPLAYED games
     from `test`. Measures "of games they own, which will they actually play".
   - cold:     per user, 1 played cold game + 99 cold games the user does not own.
   A `valid` version of the standard protocol is used for tuning.

Ownership itself is an input (the task is "given a library, predict play"), so
features such as library size may use the full library. Anything computed from
playtime may only use `train`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ART, N_NEG, SEED
from . import data

N_HARD = 9


def _sample_negatives(rng, owned_sets, users, pool, n):
    """For each user, sample n items from `pool` not in that user's owned set."""
    out = np.empty((len(users), n), dtype=np.int32)
    for r, u in enumerate(users):
        owned = owned_sets[u]
        picks = []
        while len(picks) < n:
            cand = pool[rng.integers(0, len(pool), size=2 * n)]
            cand = [c for c in cand if c not in owned and c not in picks]
            picks.extend(cand[: n - len(picks)])
        out[r] = picks
    return out


def build(force: bool = False):
    if (ART / "train.parquet").exists() and not force:
        return
    rng = np.random.default_rng(SEED)
    inter, games, _, _ = data.load()

    # integer ids
    users = np.sort(inter.user_id.unique())
    items = np.sort(inter.item_id.unique())
    uidx = pd.Series(np.arange(len(users)), index=users)
    iidx = pd.Series(np.arange(len(items)), index=items)
    df = pd.DataFrame({
        "u": uidx[inter.user_id].values.astype(np.int32),
        "i": iidx[inter.item_id].values.astype(np.int32),
        "playtime": inter.playtime_forever.values.astype(np.int64),
        "playtime_2w": inter.playtime_2weeks.values.astype(np.int64),
    })
    df["y"] = np.log1p(df.playtime).astype(np.float32)
    df["played"] = (df.playtime > 0).astype(np.int8)
    df["active"] = (df.playtime_2w > 0).astype(np.int8)
    pd.DataFrame({"user_id": users}).to_parquet(ART / "user_index.parquet")
    pd.DataFrame({"item_id": items}).to_parquet(ART / "item_index.parquet")

    # 1. cold-start games (only games with metadata, so content models can score them)
    has_meta = np.isin(items, games.item_id.values)
    meta_idx = np.flatnonzero(has_meta)
    cold_items = rng.choice(meta_idx, size=int(0.10 * len(meta_idx)), replace=False)
    is_cold = np.zeros(len(items), bool)
    is_cold[cold_items] = True
    cold = df[is_cold[df.i.values]].reset_index(drop=True)
    warm = df[~is_cold[df.i.values]].reset_index(drop=True)

    # 2. random 80/10/10 over warm pairs
    r = rng.random(len(warm))
    split = np.where(r < 0.8, "train", np.where(r < 0.9, "valid", "test"))
    for name in ["train", "valid", "test"]:
        warm[split == name].reset_index(drop=True).to_parquet(ART / f"{name}.parquet")
    cold.to_parquet(ART / "cold.parquet")
    np.save(ART / "cold_items.npy", np.sort(cold_items))

    # 3. ranking protocols
    owned = df.groupby("u").i.agg(set).to_dict()
    warm_pool = np.flatnonzero(~is_cold).astype(np.int32)
    cold_pool = np.sort(cold_items).astype(np.int32)
    va, te = warm[split == "valid"], warm[split == "test"]

    def loo(held, pool, n_neg, name):
        pos = held[held.played == 1].groupby("u").sample(1, random_state=SEED)
        pos = pos.sort_values("u")
        negs = _sample_negatives(rng, owned, pos.u.values, pool, n_neg)
        np.savez(ART / f"rank_{name}.npz", users=pos.u.values.astype(np.int32),
                 pos=pos.i.values.astype(np.int32), negs=negs)
        return len(pos)

    n_std_valid = loo(va, warm_pool, N_NEG, "valid")
    n_std_test = loo(te, warm_pool, N_NEG, "test")
    n_cold = loo(cold, cold_pool, N_NEG, "cold")

    # hard protocol: positive + 9 owned-unplayed, all from test
    unpl = te[te.played == 0]
    cnt = unpl.groupby("u").size()
    elig = set(cnt[cnt >= N_HARD].index) & set(te[te.played == 1].u)
    pos = te[(te.played == 1) & te.u.isin(elig)].groupby("u").sample(1, random_state=SEED).sort_values("u")
    hn = (unpl[unpl.u.isin(elig)].groupby("u").sample(N_HARD, random_state=SEED)
          .sort_values("u").i.values.reshape(-1, N_HARD))
    np.savez(ART / "rank_hard.npz", users=pos.u.values.astype(np.int32),
             pos=pos.i.values.astype(np.int32), negs=hn.astype(np.int32))

    stats = dict(n_users=len(users), n_items=len(items), n_cold_items=len(cold_items),
                 train=int((split == "train").sum()), valid=int((split == "valid").sum()),
                 test=int((split == "test").sum()), cold_pairs=len(cold),
                 rank_valid_users=n_std_valid, rank_test_users=n_std_test,
                 rank_hard_users=len(pos), rank_cold_users=n_cold)
    pd.Series(stats).to_json(ART / "split_stats.json")


def load(name: str) -> pd.DataFrame:
    build()
    return pd.read_parquet(ART / f"{name}.parquet")


def load_rank(name: str) -> dict:
    build()
    z = np.load(ART / f"rank_{name}.npz")
    return {k: z[k] for k in z.files}


def stats() -> dict:
    build()
    return pd.read_json(ART / "split_stats.json", typ="series").to_dict()


def leakage_checks() -> pd.DataFrame:
    """Sanity checks reported in the notebook. Every row should say True."""
    tr, va, te, co = (load(n) for n in ["train", "valid", "test", "cold"])
    key = lambda d: set(zip(d.u.values.tolist(), d.i.values.tolist()))
    ktr, kva, kte = key(tr), key(va), key(te)
    cold_items = set(np.load(ART / "cold_items.npy").tolist())
    checks = {
        "train ∩ valid pairs empty": not (ktr & kva),
        "train ∩ test pairs empty": not (ktr & kte),
        "valid ∩ test pairs empty": not (kva & kte),
        "no cold game appears in train": not (set(tr.i.unique()) & cold_items),
        "every cold pair is a cold game": set(co.i.unique()) <= cold_items,
    }
    for name in ["valid", "test", "hard", "cold"]:
        R = load_rank(name)
        pos_pairs = set(zip(R["users"].tolist(), R["pos"].tolist()))
        checks[f"rank_{name}: positives not in train"] = not (pos_pairs & ktr)
        if name == "hard":
            neg_pairs = {(u, n) for u, row in zip(R["users"].tolist(), R["negs"].tolist()) for n in row}
            checks["rank_hard: hard negatives not in train"] = not (neg_pairs & ktr)
    return pd.DataFrame({"check": list(checks), "passed": list(checks.values())})
