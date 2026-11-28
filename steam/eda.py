"""Task 2: exploratory analysis. Each function computes its numbers AND draws its figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data, features
from .viz import AQUA, BLUE, GREY, INK2, ORANGE, save


def _inter():
    inter, games, reviews, bundles = data.load()
    return inter, games, reviews, bundles


def glance() -> pd.DataFrame:
    """F3: dataset at a glance."""
    inter, games, reviews, bundles = _inter()
    n_u, n_i = inter.user_id.nunique(), inter.item_id.nunique()
    lib = inter.groupby("user_id").size()
    st = data.build()
    rows = [
        ("Users with a non-empty library", f"{n_u:,}"),
        ("Users in raw file (incl. empty libraries)", "88,310"),
        ("Games owned by at least one user", f"{n_i:,}"),
        ("User-game pairs (after de-duplication)", f"{len(inter):,}  (raw {st['raw_pairs']:,})"),
        ("Matrix density", f"{len(inter) / (n_u * n_i):.3%}"),
        ("Median games per library", f"{lib.median():.0f}  (mean {lib.mean():.0f}, max {lib.max():,})"),
        ("Owned games never played", f"{(inter.playtime_forever == 0).mean():.1%}"),
        ("Pairs played in the last 2 weeks", f"{(inter.playtime_2weeks > 0).mean():.1%}"),
        ("Owned games with store metadata", f"{st['items_with_meta']:,} of {n_i:,}"),
        ("Reviews (users / games)", f"{len(reviews):,}  ({reviews.user_id.nunique():,} / {reviews.item_id.nunique():,})"),
        ("Reviews that recommend the game", f"{reviews.recommend.mean():.1%}"),
        ("Bundles", f"{bundles.bundle_id.nunique():,}"),
    ]
    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def playtime_hist():
    """F4: raw vs log playtime, zero spike marked."""
    inter, *_ = _inter()
    pt = inter.playtime_forever.values / 60.0
    zero = (pt == 0).mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.hist(np.clip(pt, 0, 500), bins=100, color=GREY)
    ax.set_yscale("log")
    ax.set_xlabel("Hours played (clipped at 500)")
    ax.set_ylabel("User-game pairs (log scale)")
    ax.set_title("Raw hours: extremely right-skewed")
    ax = axes[1]
    y = np.log1p(inter.playtime_forever.values)
    ax.hist(y[y > 0], bins=80, color=BLUE, label="played")
    ax.bar(0, (y == 0).sum(), width=0.25, color=ORANGE, label=f"never played ({zero:.0%})")
    ax.set_xlabel("log(1 + minutes played)  =  regression target")
    ax.set_ylabel("User-game pairs")
    ax.set_title(f"{zero:.0%} of owned games are never played")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "F04_playtime_hist")
    return fig


def long_tail():
    """F5: games per user and Lorenz curve of playtime across games."""
    inter, *_ = _inter()
    lib = inter.groupby("user_id").size().values
    hours = inter.groupby("item_id").playtime_forever.sum().sort_values().values
    cum = np.concatenate([[0], np.cumsum(hours) / hours.sum()])
    x = np.linspace(0, 1, len(cum))
    gini = 1 - 2 * np.trapezoid(cum, x)
    top1 = 1 - cum[int(0.99 * len(hours))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    vals, counts = np.unique(lib, return_counts=True)
    ax.loglog(vals, counts, ".", color=BLUE, ms=4)
    ax.set_xlabel("Games in library")
    ax.set_ylabel("Number of users")
    ax.set_title(f"Libraries are long-tailed (median {np.median(lib):.0f} games)")
    ax = axes[1]
    ax.plot(x, cum, color=ORANGE)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    ax.fill_between(x, cum, x, color=ORANGE, alpha=0.12)
    ax.set_xlabel("Share of games (sorted by total playtime)")
    ax.set_ylabel("Share of all playtime")
    ax.set_title(f"Top 1% of games take {top1:.0%} of all playtime (Gini {gini:.2f})")
    fig.tight_layout()
    save(fig, "F05_long_tail")
    return fig, {"gini": gini, "top1_share": top1}


def _pairs_with_meta():
    inter, *_ = _inter()
    m = features.item_meta()[["item_id", "genres", "price", "free", "release_year", "metascore", "early_access"]]
    return inter.merge(m, on="item_id", how="left")


def pile_of_shame():
    """F6: share of owned games never played, by genre and by price band."""
    d = _pairs_with_meta()
    d["unplayed"] = d.playtime_forever == 0
    g = d[["unplayed", "genres"]].explode("genres").dropna()
    by_g = g.groupby("genres").unplayed.agg(["mean", "size"])
    by_g = by_g[by_g["size"] > 20000].sort_values("mean")
    band = pd.cut(d.price, [-0.01, 0.0, 5, 10, 20, 40, 1000],
                  labels=["Free", "<$5", "$5–10", "$10–20", "$20–40", "$40+"])
    by_p = d.groupby(band, observed=True).unplayed.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.barh(by_g.index, by_g["mean"], color=BLUE, height=0.68)
    ax.axvline(d.unplayed.mean(), color=INK2, lw=1, ls="--")
    ax.set_xlabel("Share of owned copies never played")
    ax.set_title("The 'pile of shame' varies by genre")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.bar(by_p.index.astype(str), by_p.values, color=ORANGE, width=0.68)
    ax.set_ylabel("Share never played")
    ax.set_title("Cheap games are the least played")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "F06_pile_of_shame")
    return fig, by_g, by_p


def metadata_signal():
    """F7: median playtime (played copies) by genre, free-to-play and release year."""
    d = _pairs_with_meta()
    d = d[d.playtime_forever > 0]
    d["hours"] = d.playtime_forever / 60
    g = d[["hours", "genres"]].explode("genres").dropna()
    by_g = g.groupby("genres").hours.agg(["median", "size"])
    by_g = by_g[by_g["size"] > 20000].sort_values("median")
    by_y = d[d.release_year >= 2000].groupby("release_year").hours.median()
    by_f = d.groupby(d.free.map({1.0: "Free to play", 0.0: "Paid"})).hours.median()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.4, 0.6, 1.2]})
    axes[0].barh(by_g.index, by_g["median"], color=BLUE, height=0.68)
    axes[0].set_xlabel("Median hours (played copies)")
    axes[0].set_title("Genre shifts typical playtime")
    axes[0].grid(axis="y", visible=False)
    axes[1].bar(by_f.index, by_f.values, color=[ORANGE, GREY], width=0.6)
    axes[1].set_title("Free-to-play vs paid")
    axes[1].set_ylabel("Median hours")
    axes[1].grid(axis="x", visible=False)
    axes[2].plot(by_y.index, by_y.values, color=AQUA, marker="o", ms=4)
    axes[2].set_xlabel("Release year")
    axes[2].set_ylabel("Median hours")
    axes[2].set_title("Older games: more accumulated hours")
    fig.tight_layout()
    save(fig, "F07_metadata_signal")
    return fig


def own_vs_play():
    """F8: top 15 games by owners vs by total hours."""
    inter, *_ = _inter()
    g = inter.groupby("item_name").agg(owners=("user_id", "size"), hours=("playtime_forever", "sum"))
    g["hours"] /= 60
    top_o = g.sort_values("owners", ascending=False).head(15)
    top_h = g.sort_values("hours", ascending=False).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(top_o.index[::-1], top_o.owners[::-1] / 1000, color=GREY, height=0.68)
    axes[0].set_xlabel("Owners (thousands)")
    axes[0].set_title("Most owned")
    axes[1].barh(top_h.index[::-1], top_h.hours[::-1] / 1e6, color=ORANGE, height=0.68)
    axes[1].set_xlabel("Total hours (millions)")
    axes[1].set_title("Most played")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    overlap = len(set(top_o.index) & set(top_h.index))
    fig.suptitle(f"Popular to own ≠ popular to play: only {overlap} of 15 games are in both lists",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "F08_own_vs_play")
    return fig


def reviews_view():
    """F9: recommend rate, review length vs recommend, reviewer vs non-reviewer playtime."""
    inter, _, reviews, _ = _inter()
    reviews = reviews.copy()
    reviews["words"] = reviews.review.str.split().str.len()
    rv = reviews.merge(inter[["user_id", "item_id", "playtime_forever"]], on=["user_id", "item_id"], how="left")
    reviewed = set(zip(reviews.user_id, reviews.item_id))
    key = list(zip(inter.user_id, inter.item_id))
    inter = inter.assign(reviewed=[k in reviewed for k in key])
    med = inter.groupby("reviewed").playtime_forever.median() / 60
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    rate = reviews.recommend.mean()
    axes[0].bar(["Recommend", "Not recommend"], [rate, 1 - rate], color=[BLUE, ORANGE], width=0.6)
    axes[0].set_title(f"{rate:.0%} of reviews recommend")
    axes[0].set_ylabel("Share of reviews")
    lw = reviews.groupby("recommend").words.median()
    axes[1].bar(["Recommend", "Not recommend"], [lw[True], lw[False]], color=[BLUE, ORANGE], width=0.6)
    axes[1].set_title("Negative reviews are longer")
    axes[1].set_ylabel("Median words")
    axes[2].bar(["Not reviewed", "Reviewed"], [med[False], med[True]], color=[GREY, AQUA], width=0.6)
    axes[2].set_title("Reviewed games are played far more")
    axes[2].set_ylabel("Median hours")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "F09_reviews")
    return fig, {"matched_reviews": float(rv.playtime_forever.notna().mean())}


def genre_cooccurrence():
    """Genre co-occurrence (share of games with genre A that also have genre B)."""
    G = features.genre_matrix().toarray()
    names = features.vocab()[0]
    keep = G.sum(0) > 300
    G, names = G[:, keep], [n for n, k in zip(names, keep) if k]
    C = G.T @ G
    P = C / np.diag(C)[:, None]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="P(column genre | row genre)", shrink=0.8)
    ax.set_title("Indie overlaps everything; genres are not exclusive")
    fig.tight_layout()
    save(fig, "F09b_genre_cooccurrence")
    return fig
