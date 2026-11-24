"""Task 1a: load the four raw Steam files, clean them, cache as parquet.

Outputs (in artifacts/):
  interactions.parquet  one row per (user, game) in a non-empty library
  games.parquet         one row per game with cleaned metadata
  reviews.parquet       one row per review
  bundles.parquet       one row per (bundle, game)
"""
from __future__ import annotations

import ast
import gzip
import html
import re

import numpy as np
import pandas as pd

from .config import ART, DATA


def _iter_literal(path):
    """Files are Python dict literals (not JSON), one per line."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ast.literal_eval(line)


# ---------------------------------------------------------------- interactions
def load_interactions() -> pd.DataFrame:
    rows = []
    for d in _iter_literal(DATA / "australian_users_items.json.gz"):
        u = d["user_id"]
        for it in d["items"]:
            rows.append((u, it["item_id"], it["item_name"],
                         it["playtime_forever"], it["playtime_2weeks"]))
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "item_name",
                                     "playtime_forever", "playtime_2weeks"])
    n_raw = len(df)
    # The raw file contains a few duplicated users; keep one row per pair.
    df = df.drop_duplicates(["user_id", "item_id"], keep="first").reset_index(drop=True)
    df.attrs["n_raw"] = n_raw
    return df


# ----------------------------------------------------------------------- games
def _parse_price(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return np.nan, False
    if isinstance(p, (int, float)):
        return float(p), float(p) == 0.0
    s = str(p).lower()
    if "free" in s:
        return 0.0, True
    try:
        return float(s), False
    except ValueError:
        return np.nan, False


def load_games() -> pd.DataFrame:
    rows = []
    for d in _iter_literal(DATA / "steam_games.json.gz"):
        if "id" not in d:
            continue
        price, free = _parse_price(d.get("price"))
        genres = d.get("genres") or []
        free = free or ("Free to Play" in genres)
        meta = d.get("metascore")
        try:
            meta = float(meta)
        except (TypeError, ValueError):
            meta = np.nan
        ym = re.search(r"(19|20)\d\d", str(d.get("release_date") or ""))
        rows.append(dict(
            item_id=str(d["id"]),
            title=d.get("app_name") or d.get("title"),
            genres=[html.unescape(g) for g in genres],
            tags=[html.unescape(t) for t in (d.get("tags") or [])],
            specs=list(d.get("specs") or []),
            price=price,
            free=bool(free),
            developer=d.get("developer"),
            publisher=d.get("publisher"),
            release_year=float(ym.group(0)) if ym else np.nan,
            metascore=meta,
            sentiment=d.get("sentiment"),
            early_access=bool(d.get("early_access", False)),
        ))
    g = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    return g


# --------------------------------------------------------------------- reviews
_HELP = re.compile(r"(\d[\d,]*) of (\d[\d,]*) people")
_DATE = re.compile(r"Posted (\w+ \d+)(?:, (\d{4}))?")


def _parse_posted(s):
    m = _DATE.search(s or "")
    if not m:
        return pd.NaT
    year = m.group(2) or "2015"   # reviews without a year are from the crawl year
    return pd.to_datetime(f"{m.group(1)} {year}", format="%B %d %Y", errors="coerce")


def load_reviews() -> pd.DataFrame:
    rows = []
    for d in _iter_literal(DATA / "australian_user_reviews.json.gz"):
        for r in d["reviews"]:
            m = _HELP.search(r.get("helpful", ""))
            hv, ht = (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))) if m else (0, 0)
            rows.append(dict(user_id=d["user_id"], item_id=str(r["item_id"]),
                             recommend=bool(r["recommend"]), review=r.get("review", ""),
                             posted=_parse_posted(r.get("posted")),
                             helpful_votes=hv, helpful_total=ht))
    return pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)


# --------------------------------------------------------------------- bundles
def load_bundles() -> pd.DataFrame:
    rows = []
    for b in _iter_literal(DATA / "bundle_data.json.gz"):
        for it in b.get("items", []):
            rows.append(dict(bundle_id=str(b["bundle_id"]), bundle_name=b["bundle_name"],
                             bundle_price=str(b.get("bundle_final_price")),
                             discount=str(b.get("bundle_discount")),
                             item_id=str(it["item_id"])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- build/cache
def build(force: bool = False) -> dict:
    """Parse everything once and cache. Returns a dict of cleaning stats."""
    out = ART / "interactions.parquet"
    if out.exists() and not force:
        return pd.read_json(ART / "data_stats.json", typ="series").to_dict()

    inter = load_interactions()
    games = load_games()
    reviews = load_reviews()
    bundles = load_bundles()

    stats = dict(raw_pairs=int(inter.attrs["n_raw"]), dedup_pairs=len(inter),
                 users=int(inter.user_id.nunique()), items=int(inter.item_id.nunique()),
                 games_meta=len(games),
                 items_with_meta=int(inter.item_id.drop_duplicates().isin(games.item_id).sum()),
                 reviews=len(reviews), bundles=int(bundles.bundle_id.nunique()))

    inter.to_parquet(out)
    games.to_parquet(ART / "games.parquet")
    reviews.to_parquet(ART / "reviews.parquet")
    bundles.to_parquet(ART / "bundles.parquet")
    pd.Series(stats).to_json(ART / "data_stats.json")
    return stats


def load():
    """Return (interactions, games, reviews, bundles) from cache."""
    build()
    return (pd.read_parquet(ART / "interactions.parquet"),
            pd.read_parquet(ART / "games.parquet"),
            pd.read_parquet(ART / "reviews.parquet"),
            pd.read_parquet(ART / "bundles.parquet"))
