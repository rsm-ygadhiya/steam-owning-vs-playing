"""Task 6 / 9: results tables and the modeling + evaluation figures (F1-F2, F10-F25).

Reads only cached artifacts. Never imports torch (the notebook kernel must not
load torch and LightGBM together on macOS).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

from .config import ART, SEED
from . import features, splits
from .viz import (AQUA, BLUE, GREY, INK, INK2, MAGENTA, ORANGE, SEQ, YELLOW, box_diagram, color,
                  family_legend, hbar_models, save)

RES = ART / "results"


def load(name):
    with open(RES / f"{name}.json") as f:
        d = json.load(f)
    if name == "ranking" and (RES / "ranking_ltr.json").exists():   # merge the LightGBM ranking stage
        with open(RES / "ranking_ltr.json") as f:
            for proto, rows in json.load(f)["results"].items():
                d["results"][proto].update(rows)
    return d


# ------------------------------------------------------------------ tables
def table_regression(split="test"):
    r = load("regression")["results"][split]
    df = pd.DataFrame(r).T
    base = df.loc["Global mean", "MSE"]
    df["vs global mean"] = 1 - df.MSE / base
    return df


def table_classification(target="played"):
    return pd.DataFrame(load("classification")[target]).T


def table_ranking(proto="test"):
    return pd.DataFrame(load("ranking")["results"][proto]).T


def bootstrap_regression(pairs=(("Ensemble", "LightGBM (stacked)"), ("Hurdle (LightGBM)", "LightGBM (stacked)"),
                                ("LightGBM (stacked)", "Latent factor (MF)"), ("Latent factor (MF)", "Bias model"),
                                ("Bias model", "Item mean")), n_boot=500):
    """Paired bootstrap over test pairs: 95% CI of MSE(A) - MSE(B). Negative = A is better."""
    te = splits.load("test")
    P = pd.read_parquet(ART / "reg_preds_test.parquet")
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(te), len(te)) for _ in range(n_boot)]
    rows = []
    for a, b in pairs:
        d = (P[a].values - te.y.values) ** 2 - (P[b].values - te.y.values) ** 2
        means = np.array([d[i].mean() for i in idx])
        lo, hi = np.quantile(means, [0.025, 0.975])
        rows.append({"A": a, "B": b, "ΔMSE (A − B)": d.mean(), "95% CI low": lo, "95% CI high": hi,
                     "significant": "yes" if hi < 0 or lo > 0 else "no"})
    return pd.DataFrame(rows)


def table_ltr_significance():
    """Paired bootstrap over test users: LTR reranker NDCG@10 minus each base model."""
    with open(RES / "ranking_ltr.json") as f:
        p = json.load(f)["paired_ndcg10_test"]
    df = pd.DataFrame(p).T.rename(columns={"diff": "ΔNDCG@10 (LTR − model)", "lo": "95% CI low", "hi": "95% CI high"})
    df["significant"] = np.where((df["95% CI low"] > 0) | (df["95% CI high"] < 0), "yes", "no")
    return df


def ltr_importance(k=12):
    with open(RES / "ranking_ltr.json") as f:
        imp = pd.Series(json.load(f)["importance"])
    return (imp / imp.sum()).sort_values(ascending=False).head(k).rename("share of gain").to_frame()


def style_table(df, best_low=(), best_high=(), fmt="{:.4f}"):
    s = df.style.format(fmt)
    for c in best_low:
        if c in df:
            s = s.highlight_min(subset=[c], props="font-weight:bold; background-color:#e2edfb")
    for c in best_high:
        if c in df:
            s = s.highlight_max(subset=[c], props="font-weight:bold; background-color:#e2edfb")
    return s


# --------------------------------------------------------------- diagrams
def fig_pipeline():
    """F1: the whole project in one picture."""
    fig, ax = plt.subplots(figsize=(12, 3.4))
    L = "#e2edfb"
    box_diagram(ax, [
        (1, 14, 17, 12, "User's Steam\nlibrary\n(+ game metadata)", "#efeee9"),
        (25, 26, 20, 11, "Task 4\nWill they play it?\n(AUC / BER)", L),
        (25, 3, 20, 11, "Task 3\nHow long?\nlog(1+minutes)  (MSE)", L),
        (53, 14, 20, 12, "Hurdle model\nP(play) × E[time | play]", ORANGE),
        (80, 14, 19, 12, "Task 5\nRecommend games\nthey will PLAY\n(HR@10, NDCG)", BLUE),
    ], [((18.5, 20), (24.5, 31)), ((18.5, 20), (24.5, 8.5)), ((45.5, 31), (52.5, 22)),
        ((45.5, 8.5), (52.5, 18)), ((73.5, 20), (79.5, 20))])
    ax.set_title("Owning a game isn't the same as playing it: three connected predictive tasks")
    save(fig, "F01_pipeline")
    return fig


def table_tasks():
    """F2: task x metric x baseline."""
    return pd.DataFrame([
        ["3 (main)", "Hours played", "Regression on y = log(1 + minutes) for owned pairs",
         "MSE (primary), MAE", "Global / user / game mean, bias model"],
        ["4", "Will they play it?", "Binary: playtime > 0 (also: active in last 2 weeks)",
         "AUC, BER (threshold set on valid)", "Game / user play-rate, logistic regression"],
        ["5", "Recommend games they'll play", "Ranking over games; positive = played",
         "AUC, HR@10, NDCG@10; coverage, novelty", "Random, most-popular, Jaccard item-kNN"],
    ], columns=["Task", "Question", "Formulation", "Metric", "Baselines to beat"])


def fig_ladder():
    """F10: model ladder."""
    fig, ax = plt.subplots(figsize=(12.5, 2.6))
    steps = [("Means", GREY), ("Bias\nα+β_u+β_i", BLUE), ("Latent factors\n+ γ_u·γ_i", BLUE),
             ("Factorization\nmachine\n+ metadata", ORANGE), ("LightGBM\nstacked", ORANGE),
             ("Hurdle\nP×E", ORANGE), ("CatBoost\n+ MLP", ORANGE), ("Non-neg.\nensemble", ORANGE)]
    w, gap = 10.6, 1.9
    boxes, arrows = [], []
    for k, (t, c) in enumerate(steps):
        x = 1 + k * (w + gap)
        boxes.append((x, 12, w, 15, t, c))
        if k:
            arrows.append(((x - gap + 0.2, 19.5), (x - 0.2, 19.5)))
    box_diagram(ax, boxes, arrows)
    ax.set_title("Model ladder: each step adds one idea (grey baseline, blue course model, orange advanced)")
    save(fig, "F10_ladder")
    return fig


def fig_hurdle():
    """F11: hurdle schematic."""
    fig, ax = plt.subplots(figsize=(11, 3))
    box_diagram(ax, [
        (2, 14, 18, 12, "(user, game)\nfeatures", "#efeee9"),
        (30, 25, 26, 11, "Stage 1: classifier\np = P(played)", "#e2edfb"),
        (30, 3, 26, 11, "Stage 2: regressor on\nplayed pairs only\nμ = E[y | played]", "#e2edfb"),
        (68, 14, 29, 12, "ŷ = p · μ\n(MSE-optimal, since\ny = 0 when unplayed)", ORANGE),
    ], [((20.5, 20), (29.5, 30)), ((20.5, 20), (29.5, 8.5)), ((56.5, 30), (67.5, 22)), ((56.5, 8.5), (67.5, 18))])
    ax.set_title("Hurdle model: separate 'do they start?' from 'how long do they keep going?'")
    save(fig, "F11_hurdle")
    return fig


# ---------------------------------------------------------------- modeling
def fig_training_curves():
    """F12: train vs valid MSE per epoch for MF and FM."""
    t = load("regression_cf")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, key, name in [(axes[0], "mf_curve", "Latent factor model"), (axes[1], "fm_curve", "Factorization machine")]:
        h = pd.DataFrame(t[key])
        ax.plot(h.epoch, h.train_mse, color=GREY, marker="o", ms=4, label="train")
        ax.plot(h.epoch, h.valid_mse, color=BLUE if key == "mf_curve" else ORANGE, marker="o", ms=4, label="valid")
        b = h.valid_mse.idxmin()
        ax.annotate(f"best valid {h.valid_mse[b]:.3f}\n(epoch {h.epoch[b]})", (h.epoch[b], h.valid_mse[b]),
                    xytext=(10, 25), textcoords="offset points", fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel("MSE")
    fig.suptitle("Correctly scaled L2 keeps MF and FM stable: valid MSE no longer blows up after the best epoch",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "F12_training_curves")
    return fig


def fig_tsne(n=2500):
    """F13: t-SNE of ALS item factors, colored by a game's leading tag."""
    V = np.load(ART / "als_item_factors.npy")
    tr = splits.load("train")
    pop = np.bincount(tr[tr.played == 1].i.values, minlength=len(V))
    top = np.argsort(-pop)[:n]
    m = features.item_meta()
    groups = [("FPS", BLUE), ("Strategy", ORANGE), ("RPG", AQUA)]

    def lab(tags):
        for t in tags:
            for g, _ in groups:
                if t == g:
                    return g
        return "Other"
    labels = np.array([lab(m.tags.values[i]) for i in top])
    Vn = V[top] / (np.linalg.norm(V[top], axis=1, keepdims=True) + 1e-9)
    Z = TSNE(2, metric="cosine", init="pca", random_state=SEED, perplexity=30).fit_transform(Vn)
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    o = labels == "Other"
    ax.scatter(Z[o, 0], Z[o, 1], s=6, color="#d8d7d1", label="Other", linewidths=0)
    for g, c in groups:
        k = labels == g
        ax.scatter(Z[k, 0], Z[k, 1], s=12, color=c, label=g, edgecolors="white", linewidths=0.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(loc="upper right", markerscale=2)
    ax.set_title("ALS embeddings (no tags used): FPS games form tight clusters; strategy and RPG overlap more")
    save(fig, "F13_tsne")
    return fig


def fig_importance(k=15):
    """F14: LightGBM gain importance."""
    imp = pd.read_json(RES / "lgb_importance.json", typ="series")
    imp = (imp / imp.sum()).head(k)
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = [ORANGE if c.endswith(("pred", "dot", "i2v_sim")) else BLUE for c in imp.index]
    ax.barh(imp.index[::-1], imp.values[::-1], color=cols[::-1], height=0.68)
    ax.set_xlabel("Share of total gain")
    ax.grid(axis="y", visible=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              ["model output (stacked)", "engineered feature"], loc="lower right")
    ax.set_title("The stack leans on the CF model outputs; engineered features refine at the margin")
    save(fig, "F14_importance")
    return fig


# -------------------------------------------------------------- evaluation
def fig_calibration(model="Ensemble", n_bins=20):
    """F26: are the predictions trustworthy? Regression and classifier calibration on TEST."""
    te = splits.load("test")
    p = pd.read_parquet(ART / "reg_preds_test.parquet")[model].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": te.y.values, "q": q}).groupby("q")
    lo, hi = g.y.quantile(0.25), g.y.quantile(0.75)
    ax.fill_between(g.p.mean(), lo, hi, color=BLUE, alpha=0.15, label="actual: middle 50%")
    ax.plot(g.p.mean(), g.y.mean(), "o-", color=BLUE, ms=4, label="actual: mean")
    ax.plot([0, 9], [0, 9], ls="--", color=GREY, lw=1, label="perfect calibration")
    ax.set_xlabel(f"Predicted log(1 + minutes)  ({model}, {n_bins} bins)")
    ax.set_ylabel("Actual log(1 + minutes)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Regression: unbiased in every bin, but wide spread per pair", fontsize=11)
    ax = axes[1]
    C = pd.read_parquet(ART / "cls_probs_played.parquet")
    for name, c in [("Logistic (features)", BLUE), ("LightGBM (stacked)", ORANGE)]:
        q = pd.qcut(C[name], 10, labels=False, duplicates="drop")
        g = C.groupby(q)
        ax.plot(g[name].mean(), g.y.mean(), "o-", color=c, ms=4, label=name)
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.set_xlabel("Predicted P(played)  (deciles)")
    ax.set_ylabel("Observed share played")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Classifier: predicted probabilities match observed rates", fontsize=11)
    fig.suptitle("Validity check: predictions are calibrated on held-out test data",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "F26_calibration")
    return fig


def calibration_stats(model="Ensemble"):
    """Rank agreement overall and within each user's own library (test)."""
    te = splits.load("test")
    p = pd.read_parquet(ART / "reg_preds_test.parquet")[model].values
    d = pd.DataFrame({"u": te.u.values, "p": p, "y": te.y.values})
    big = d[d.groupby("u").u.transform("size") >= 5]
    r = big.groupby("u")[["p", "y"]].rank()
    r["u"] = big.u.values
    within = r.groupby("u").apply(lambda x: x.p.corr(x.y)).dropna()
    rmse = float(np.sqrt(np.mean((p - te.y.values) ** 2)))
    return pd.Series({"RMSE (log scale)": rmse, "typical error factor e^RMSE": float(np.exp(rmse)),
                      "Spearman, all test pairs": d.p.corr(d.y, method="spearman"),
                      "median within-user Spearman (≥5 test games)": within.median(),
                      "users in within-user check": len(within)}).to_frame("value")


def table_lgb_tuning():
    df = pd.DataFrame(load("lgb_tuning"))
    df["gap"] = df.oof_mse - df.train_mse
    return df.set_index(["model", "setting"])[["learning_rate", "num_leaves", "min_data_in_leaf", "lambda_l2",
                                               "feature_fraction", "trees", "train_mse", "oof_mse", "gap"]]


def table_fit_gap():
    """Train vs out-of-fold MSE for every cross-fit learner: a small gap means no overfitting."""
    g = load("regression")["tuning"].get("fit_gap", {})
    df = pd.DataFrame(g).T
    if (RES / "regression_nn.json").exists():
        f = pd.DataFrame(load("regression_nn")["folds"])
        df.loc["MLP (neural net)"] = [f.train_mse.mean(), f.oof_mse.mean(), f.epochs.mean()]
    df["gap (oof − train)"] = df.oof_mse - df.train_mse
    return df.rename(columns={"rounds": "trees / epochs"})


def fig_mse_bars(split="test"):
    """F16: MSE by model."""
    df = table_regression(split).sort_values("MSE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    hbar_models(ax, list(df.index), list(df.MSE), ref=df.loc["Global mean", "MSE"], ref_label="global mean")
    ax.set_xlabel(f"{split.capitalize()} MSE on log(1 + minutes)   (lower is better)")
    family_legend(ax)
    best = df.MSE.idxmin()
    ax.set_title(f"{best} cuts error by {df.loc[best, 'vs global mean']:.0%} vs predicting the mean")
    save(fig, f"F16_mse_{split}")
    return fig


def fig_tuning_heatmap():
    """F17: valid MSE over MF K x lambda."""
    t = pd.DataFrame(load("regression_cf")["mf"])
    H = t.pivot(index="K", columns="lambda", values="valid_mse")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(H.values, cmap="Blues_r", aspect="auto")
    ax.set_xticks(range(H.shape[1]), [f"{c:g}" for c in H.columns])
    ax.set_yticks(range(H.shape[0]), H.index)
    ax.set_xlabel("λ (L2 strength)")
    ax.set_ylabel("K (latent dims)")
    ax.grid(False)
    for (a, b), v in np.ndenumerate(H.values):
        ax.text(b, a, f"{v:.3f}", ha="center", va="center", fontsize=9,
                color="white" if v < H.values.mean() else INK)
    fig.colorbar(im, ax=ax, label="valid MSE", shrink=0.85)
    ax.set_title("Tuned on validation: the L2 scale matters more than K")
    save(fig, "F17_tuning")
    return fig


def fig_breakdown(bd: pd.DataFrame, models=("Item mean", "Bias model", "Latent factor (MF)", "Ensemble")):
    """F18: MSE by user activity and game popularity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    pal = [GREY, BLUE, AQUA, ORANGE]
    for ax, by in zip(axes, bd.by.unique()):
        d = bd[(bd.by == by) & bd.model.isin(models)]
        bins = list(dict.fromkeys(d.bin))
        w = 0.8 / len(models)
        for k, (mname, c) in enumerate(zip(models, pal)):
            v = d[d.model == mname].set_index("bin").reindex(bins).MSE
            ax.bar(np.arange(len(bins)) + (k - (len(models) - 1) / 2) * w, v.values, w * 0.92, color=c, label=mname)
        ax.set_xticks(range(len(bins)), bins)
        ax.set_xlabel(by)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Test MSE")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("Users with ≤10 known games are hardest; latent factors help most for heavy users",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "F18_breakdown")
    return fig


def fig_ranking():
    """F19: ranking quality on the three protocols."""
    R = load("ranking")
    seeds = R["seeds"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"wspace": 1.05})
    specs = [("test", "NDCG@10", "Standard: played vs 99 not-owned"),
             ("hard", "AUC", "Hard: played vs 9 owned-but-unplayed"),
             ("cold", "NDCG@10", "Cold start: never-seen games")]
    for ax, (proto, met, title) in zip(axes, specs):
        df = pd.DataFrame(R["results"][proto]).T.sort_values(met, ascending=False)
        errs = [np.std(seeds[n][met]) if (proto == "test" and n in seeds) else 0 for n in df.index]
        hbar_models(ax, list(df.index), list(df[met]), errs=errs)
        ax.set_xlabel(met)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Popularity is a strong ranker until the task asks which OWNED games get played",
                 x=0.01, y=1.04, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "F19_ranking")
    return fig


def fig_roc():
    """F20: ROC curves for played / active."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, target in zip(axes, ["played", "active"]):
        P = pd.read_parquet(ART / f"cls_probs_{target}.parquet")
        y = P.pop("y").values
        for name, c in [("Item rate (baseline)", GREY), ("Logistic (features)", BLUE),
                        ("Logistic (+ CF scores)", AQUA), ("LightGBM (stacked)", ORANGE)]:
            fpr, tpr, _ = roc_curve(y, P[name].values)
            auc = load("classification")[target][name]["AUC"]
            ax.plot(fpr, tpr, color=c, label=f"{name}  AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title({"played": "Played at all (64% positive)", "active": "Played in last 2 weeks (3% positive)"}[target],
                     fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Adding collaborative-filtering scores is the big jump in 'will they play?'",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "F20_roc")
    return fig


def ablations() -> pd.DataFrame:
    """F21 data: marginal value of each component (all numbers on TEST)."""
    reg = load("regression")["results"]
    rk = load("ranking")["results"]
    rows = [
        ("Regression", "Full ensemble", "–", reg["test"]["Ensemble"]["MSE"], "MSE"),
        ("Regression", "− hurdle stage (plain stacked LightGBM)", "hurdle",
         reg["test"]["LightGBM (stacked)"]["MSE"], "MSE"),
        ("Regression", "− collaborative signal (metadata-only LightGBM)", "CF",
         reg["test"]["LightGBM (metadata only)"]["MSE"], "MSE"),
        ("Regression", "− metadata & stacking (MF alone)", "metadata",
         reg["test"]["Latent factor (MF)"]["MSE"], "MSE"),
        ("Ranking (hard)", "ALS playtime-weighted", "–", rk["hard"]["ALS (playtime-weighted)"]["AUC"], "AUC"),
        ("Ranking (hard)", "− playtime weighting (binary ALS)", "playtime",
         rk["hard"]["ALS (binary)"]["AUC"], "AUC"),
        ("Ranking (hard)", "+ Task 4 played-classifier instead", "classifier",
         rk["hard"]["Played classifier (Task 4 LightGBM)"]["AUC"], "AUC"),
        ("Ranking (standard)", "ALS playtime-weighted", "–", rk["test"]["ALS (playtime-weighted)"]["NDCG@10"], "NDCG@10"),
        ("Ranking (standard)", "− playtime weighting (binary ALS)", "playtime",
         rk["test"]["ALS (binary)"]["NDCG@10"], "NDCG@10"),
        ("Ranking (standard)", "+ content features (hybrid)", "content",
         rk["test"]["Hybrid (ALS + content)"]["NDCG@10"], "NDCG@10"),
        ("Ranking (standard)", "+ every model combined (LTR reranker)", "LTR",
         rk["test"]["LTR reranker (LightGBM)"]["NDCG@10"], "NDCG@10"),
    ]
    df = pd.DataFrame(rows, columns=["task", "variant", "component", "value", "metric"])
    df["Δ vs full"] = df.groupby("task").value.transform(lambda v: v - v.iloc[0])
    return df


def fig_ablation():
    df = ablations()
    reg = df[df.task == "Regression"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 1.1})
    ax = axes[0]
    d = reg.iloc[1:]
    ax.barh(d.variant[::-1], d["Δ vs full"][::-1], color=ORANGE, height=0.6)
    for y, v in enumerate(d["Δ vs full"][::-1]):
        ax.text(v, y, f"  +{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, d["Δ vs full"].max() * 1.25)
    ax.set_xlabel("Increase in test MSE when the component is removed")
    ax.set_title("Regression: removing collaborative signal hurts most", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    rk = df[df.task != "Regression"]
    d = rk[rk.component != "–"]
    lab = [f"{t.split('(')[1][:-1]}: {v}" for t, v in zip(d.task, d.variant)]
    ax.barh(lab[::-1], d["Δ vs full"][::-1], color=[BLUE if x < 0 else AQUA for x in d["Δ vs full"][::-1]], height=0.6)
    for y, v in enumerate(d["Δ vs full"][::-1]):
        ax.text(max(v, 0), y, f"  {v:+.4f}", va="center", ha="left", fontsize=9, color=INK2)   # negatives labelled right of 0
    lo, hi = d["Δ vs full"].min(), d["Δ vs full"].max()
    ax.set_xlim(lo * 1.45 if lo < 0 else 0, hi * 1.6 if hi > 0 else 0.005)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Change in metric vs playtime-weighted ALS")
    ax.set_title("Ranking: playtime weighting helps only on the hard protocol;\ncombining models and the Task 4 classifier help most", fontsize=11)
    ax.grid(axis="y", visible=False)
    save(fig, "F21_ablation")
    return fig


def fig_popularity_bias():
    """F22: head share and coverage of top-10 recommendations."""
    b = pd.DataFrame(load("ranking")["beyond"]).T
    rk = pd.DataFrame(load("ranking")["results"]["test"]).T
    b = b.sort_values("head_share")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"wspace": 0.7})
    ax = axes[0]
    y = np.arange(len(b))
    TAIL = "#c9c8c1"
    ax.barh(y, b.head_share, color=ORANGE, height=0.66)
    ax.barh(y, 1 - b.head_share, left=b.head_share + 0.004, color=TAIL, height=0.66)
    ax.set_yticks(y, b.index)
    ax.set_xlim(0, 1.005)
    ax.set_xlabel("Share of top-10 recommendation slots")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=TAIL)],
              ["top-1% most-played games", "the other 99%"], loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Who gets recommended?", fontsize=11)
    ax = axes[1]
    for n in b.index:
        ax.scatter(rk.loc[n, "NDCG@10"], b.loc[n, "coverage"], s=60, color=color(n), edgecolors="white", zorder=3)
        off = {"ALS (binary)": (6, -13), "Hybrid (ALS + content)": (6, -12)}.get(n, (6, 3))
        ax.annotate(n, (rk.loc[n, "NDCG@10"], b.loc[n, "coverage"]), xytext=off,
                    textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlabel("Accuracy (standard NDCG@10)")
    ax.set_ylabel("Catalog coverage (share of games ever recommended)")
    ax.set_title("Accuracy vs. diversity trade-off", fontsize=11)
    fig.suptitle("Accuracy vs exposure: popularity and kNN push hits; BPR and content spread recommendations wider",
                 x=0.01, y=1.04, ha="left", fontsize=12.5, fontweight="bold")
    save(fig, "F22_popularity_bias")
    return fig


def fig_errors(model="Ensemble", n_worst=8):
    """F23: predicted vs actual + worst errors."""
    te = splits.load("test")
    P = pd.read_parquet(ART / "reg_preds_test.parquet")
    p = P[model].values
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    hb = ax.hexbin(te.y.values, p, gridsize=60, bins="log", cmap="Blues", mincnt=1)
    ax.plot([0, 12], [0, 12], color=ORANGE, lw=1.5, ls="--")
    ax.set_xlabel("Actual log(1 + minutes)")
    ax.set_ylabel(f"Predicted ({model})")
    ax.grid(False)
    fig.colorbar(hb, ax=ax, label="pairs (log)")
    ax.set_title("The model hedges toward the middle on zero-playtime pairs")
    save(fig, "F23_errors")
    m = features.item_meta()
    items = pd.read_parquet(ART / "item_index.parquet").item_id.values
    err = (p - te.y.values)
    idx = np.argsort(-np.abs(err))[:n_worst]
    worst = pd.DataFrame({
        "game": [m.title.values[i] if isinstance(m.title.values[i], str) else f"(no metadata) id {items[i]}" for i in te.i.values[idx]],
        "actual hours": (np.expm1(te.y.values[idx]) / 60).round(1),
        "predicted hours": (np.expm1(p[idx]) / 60).round(1),
        "free": m.free.values[te.i.values[idx]],
        "user library size": features.train_stats()[0].library_size.values[te.u.values[idx]],
    })
    return fig, worst


def user_card(seed=3):
    """F24: one user's library and ALS recommendations with 'because you played X'."""
    z = np.load(ART / "rank_top10.npz")
    users = z["users"]
    top = z["ALS_(playtime-weighted)"]
    V = np.load(ART / "als_item_factors.npy")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    tr = splits.load("train")
    m = features.item_meta()
    names = m.title.fillna("(unknown)").values
    rng = np.random.default_rng(seed)
    lib_sizes = tr[tr.played == 1].groupby("u").size()
    cand = [u for u in users if 15 <= lib_sizes.get(u, 0) <= 40]
    u = cand[rng.integers(len(cand))]
    k = int(np.flatnonzero(users == u)[0])
    hist = tr[(tr.u == u) & (tr.played == 1)].sort_values("playtime", ascending=False)
    played = hist.i.values[:15]
    lib = pd.DataFrame({"most played games": names[hist.i.values[:6]],
                        "hours": (hist.playtime.values[:6] / 60).round(0).astype(int)})
    rows = []
    for j in top[k][:5]:
        sims = Vn[played] @ Vn[j]
        b = played[int(np.argmax(sims))]
        rows.append({"recommended": names[j], "genres": ", ".join(m.genres.values[j][:3]),
                     "because you played": names[b], "similarity": round(float(sims.max()), 2)})
    return lib, pd.DataFrame(rows)


def table_related_work():
    """F25."""
    return pd.DataFrame([
        ["Kang & McAuley 2018 (SASRec, ICDM)", "Steam (this crawl, larger version)", "Next-item prediction",
         "Self-attentive sequential model", "HR@10 / NDCG@10, 100 sampled negatives",
         "Same sampled protocol. We rank by library (no order); SASRec reports most-popular is a weak baseline, but here popularity is strong because libraries overlap heavily"],
        ["Wan & McAuley 2018 (RecSys)", "Steam (this dataset)", "Monotonic behavior chains (purchase → play → review)",
         "Chain-aware BPR variants", "AUC", "Our 'owned vs played' split is one link of their chain; we confirm play signal is separate from ownership"],
        ["Pathak, Gupta & McAuley 2017 (SIGIR)", "Steam bundles", "Bundle recommendation", "BPR + bundle generation",
         "AUC, recall", "We use bundles only for EDA; bundle scoring is a planned extension"],
        ["Koren, Bell & Volinsky 2009", "Netflix", "Rating prediction", "Biased matrix factorization", "RMSE",
         "Our bias → MF ladder mirrors theirs: biases capture most of the gain, factors add more"],
        ["Hu, Koren & Volinsky 2008", "TV viewing", "Implicit feedback", "Confidence-weighted ALS", "Expected rank",
         "Playtime as confidence helps on our hard protocol (owned-but-unplayed negatives) but not on the standard one"],
        ["Rendle 2009 (BPR)", "Rossmann, Netflix", "Implicit ranking", "Pairwise BPR loss", "AUC",
         "BPR beats popularity on AUC in their paper; here, even after tuning, BPR trails popularity, ALS and item2vec"],
        ["Rendle 2010 (FM)", "Various", "Sparse regression", "Factorization machines", "RMSE",
         "FM adds metadata; helps cold start more than warm accuracy"],
        ["Barkan & Koenigstein 2016 (item2vec)", "Microsoft Store, music", "Item similarity",
         "Skip-gram on item sets", "Similarity quality", "Our item2vec orders libraries by playtime, as in HW4; best single recommender on the hard protocol"],
        ["Burges 2010 (LambdaMART)", "Web search (LETOR)", "Learning to rank", "Gradient-boosted trees with lambda gradients",
         "NDCG", "Our LambdaRank reranker over all base recommenders is the best model on the standard protocol"],
    ], columns=["Work", "Data", "Task", "Method", "Metric", "How we compare"])
