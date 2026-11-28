"""Generate notebooks/workbook.ipynb from the cells below, execute it, export workbook.html.

    uv run python scripts/build_notebook.py            # build + execute + export
    uv run python scripts/build_notebook.py --no-exec  # just write the .ipynb

Edit the narrative HERE (not in the .ipynb) so the notebook stays reproducible.
Heavy computation lives in steam/ and is cached in artifacts/; the notebook
re-runs only what is missing, then reads results and draws every figure.
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "workbook.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# =============================================================================
md(r"""
# Owning ≠ Playing: Predicting What Steam Users Actually Play
**MGTA 461 / CSE 158-258, Assignment 2**

Steam users buy far more games than they play. Given a user's library, we predict
1. **how long** they play each game they own (main task, regression),
2. **whether** they play it at all, or are playing it right now (classification), and
3. **which new games** to recommend that they would actually *play*, not just buy (ranking).

All three tasks share **one data pipeline, one leak-free split and one evaluation module**.

| Section | Rubric part |
|---|---|
| [1. Predictive task](#1.-Predictive-task) | task, evaluation, baselines, validity |
| [2. Data & EDA](#2.-Data,-preprocessing-and-exploratory-analysis) | source, preprocessing, exploration |
| [3. Modeling](#3.-Modeling) | formulation, model ladder, code walkthrough, trade-offs |
| [4. Evaluation](#4.-Evaluation) | metrics, baselines, results, ablations, bias, errors |
| [5. Related work](#5.-Related-work) | prior use of this data and these methods |

*How this notebook runs:* the heavy lifting lives in the `steam/` package (one module per task).
Each stage caches its outputs in `artifacts/`, so this notebook re-runs only stages
whose outputs are missing, then reads the results and draws every figure. Model code is
printed inline (`show_code`) so you can follow the walkthrough without opening other files.
""")

code(r"""
import ast, os, subprocess, sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Markdown, display
from steam import viz, eda, report, splits, features
viz.setup()
pd.set_option("display.max_colwidth", 120)

def show_code(path, *names):
    # Print selected classes/functions from a source file (the single source of truth)
    src = (ROOT / path).read_text(); lines = src.splitlines()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            seg = "\n".join(lines[node.lineno - 1: node.end_lineno])
            display(Markdown(f"`{path}` → `{node.name}`\n```python\n{seg}\n```"))

# run any pipeline stage whose cached output is missing (each stage in its own process)
out = subprocess.run([sys.executable, "scripts/run_all.py"], capture_output=True, text=True)
print(out.stdout[-2000:], out.stderr[-2000:] if out.returncode else "")
""")

# =============================================================================
md(r"""
# 1. Predictive task

**Core story:** *owning a game isn't the same as playing it.* Over a third of the games in
Steam libraries have never been launched. A store that recommends games people will **play**
(not just buy) needs to model engagement, not ownership.
""")
code(r"""report.fig_pipeline(); plt.show()""")
code(r"""report.table_tasks()""")
md(r"""
**Why these targets and metrics**
- **Regression on `log(1 + minutes)`.** Playtime spans 0 to 600,000+ minutes, and raw-scale MSE would be dominated by
  a handful of idle games left running. On the log scale an error means "off by a factor", which is what matters for a store.
  MSE is the primary metric (it is what the course models optimize); MAE is reported for robustness.
- **Classification.** Classes are 64/36 (played) and 97/3 (active), so we report **AUC** (threshold-free) and **BER**
  (balanced error, threshold chosen on validation only), not accuracy.
- **Ranking.** Leave-one-out with sampled negatives (**HR@10, NDCG@10, AUC**), the protocol used by the dataset's
  own SASRec paper, plus a *hard* protocol whose negatives are owned-but-unplayed games, plus **coverage / novelty**,
  because accuracy alone rewards recommending only blockbusters.

**How we assess validity**
1. One fixed-seed split shared by every model; `test` is touched once, after all tuning on `valid`.
2. Explicit leakage rules (below), verified by automated checks.
3. Every model is compared against trivial baselines, and every component is ablated (Section 4).
4. Separate **cold-start** evaluation on games never seen in training.
""")
md(r"""
### The shared split (and why it is leak-free)
- **Cold-start games:** 10% of games with metadata are removed from *every* training set; all their pairs form the `cold` test set.
- The remaining pairs are split **80 / 10 / 10** into `train / valid / test`. All base models are fit on `train` only.
- **Stacked models** (LightGBM, hurdle, ensemble, logistic) are fit on `valid` with **5-fold cross-fitting**: they see
  `train`-derived features and are scored out-of-fold, so no model sees its own target.
- **Ranking protocols** are built from the same held-out rows: the positive (and every hard negative) is a held-out pair.
- **Ownership is an input, playtime is the target.** Library size may use the full library; anything computed from playtime
  (user/game averages, genre affinity, latent factors) uses `train` only. For the `active` task, no feature uses the row's own playtime.
- **Negatives are defined once:** standard = 99 warm games the user does *not* own (uniform); hard = 9 owned-but-unplayed
  held-out games; cold = 99 never-seen games the user does not own.
""")
code(r"""show_code("steam/splits.py", "build")""")
code(r"""
display(pd.Series(splits.stats(), name="count").to_frame())
splits.leakage_checks()
""")

# =============================================================================
md(r"""
# 2. Data, preprocessing and exploratory analysis

**Source.** The *Steam* datasets from the McAuley lab at UCSD
([datasets page](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)), crawled from public Steam community
profiles of **Australian users** (Kang & McAuley 2018; Wan & McAuley 2018; Pathak et al. 2017).

| File | Content |
|---|---|
| `australian_users_items` | each user's full library with `playtime_forever` and `playtime_2weeks` (minutes) |
| `australian_user_reviews` | review text, recommend flag, date, helpful votes |
| `steam_games` | store metadata: genres, tags, specs, price, developer, release date, metascore, early access |
| `bundle_data` | bundles, their games and discounts |

**Preprocessing** (`steam/data.py`):
1. Files are Python literals, not JSON, so they are parsed with `ast.literal_eval`.
2. **Empty libraries are dropped** (17,398 of 88,310 users), and **duplicated (user, game) rows are removed** (the raw file repeats some users).
3. Prices like `"Free To Play"` become 0 with a `free` flag; release year is extracted by regex; HTML entities in genres/tags are unescaped.
4. Review dates without a year are assigned the crawl year; helpful votes are parsed from text.
5. User and game IDs are mapped to contiguous integers; games without store metadata (16%) are kept, with missing features.
""")
code(r"""show_code("steam/data.py", "load_interactions", "_parse_price")""")
code(r"""eda.glance()""")
md(r"""### Playtime: why we log-transform, and why there is a zero spike""")
code(r"""eda.playtime_hist(); plt.show()""")
md(r"""
The raw distribution is so skewed that MSE on raw minutes would be dominated by a few extreme pairs. After `log(1+x)`
the played pairs look roughly bell-shaped, **but a third of all mass sits exactly at zero**. That point mass is the
motivation for the **hurdle model** in Section 3: one stage for "did they start?", another for "how long?".
""")
code(r"""fig, lt = eda.long_tail(); plt.show(); lt""")
md(r"""
Both users and games are long-tailed. A tiny fraction of games absorbs most playtime, so a **popularity baseline will be strong**,
and we must check whether our recommenders do anything beyond pushing hits (Section 4, popularity bias).
""")
code(r"""fig, by_genre, by_price = eda.pile_of_shame(); plt.show()""")
md(r"""
The "pile of shame" is not uniform. Cheap games (often bought in bundles or sales) are the least played, while expensive
games are usually played. Genre matters too. **Metadata carries signal about engagement**, which justifies the
metadata-aware models (factorization machine, LightGBM, content-based cold start).
""")
code(r"""eda.metadata_signal(); plt.show()""")
code(r"""eda.own_vs_play(); plt.show()""")
md(r"""
Ownership and play diverge: free games and test builds (e.g. *Dota 2 Test*) are owned by everyone, while a few games
(CS:GO, Garry's Mod) absorb enormous hours. **A recommender trained on ownership alone would learn the wrong thing.**
""")
code(r"""fig, rv = eda.reviews_view(); plt.show(); rv""")
code(r"""eda.genre_cooccurrence(); plt.show()""")
md(r"""
Reviews cover under 1.2% of pairs and are 88% positive, so they are too sparse and too skewed to be the main signal.
They confirm that reviewed games are heavily played. Genres overlap heavily (Indie co-occurs with most genres), so we
encode them as multi-hot features rather than a single category.
""")

# =============================================================================
md(r"""
# 3. Modeling

### Formulation
| Task | Input | Output | Optimized |
|---|---|---|---|
| 3: hours | (user, game) in the user's library + metadata | ŷ ≈ log(1 + minutes) | squared error + L2 |
| 4: will play | same | P(played), P(active) | log-loss |
| 5: recommend | user's training history | a score per game; rank | ALS weighted squared loss / BPR pairwise loss / skip-gram |

Each model on the ladder adds **one idea** to the one before, so improvements can be attributed.
""")
code(r"""report.fig_ladder(); plt.show()""")
md(r"""
### 3.1 Bias model: `ŷ = α + β_u + β_i` (course week 4)
Fit by exact alternating updates. Each step solves its regularized least-squares subproblem in closed form, using `bincount`
over 3.8M rows (a few seconds). λ is tuned on `valid`.
""")
code(r"""show_code("steam/regression_cf.py", "BiasModel")""")
md(r"""
### 3.2 Latent factor model: `ŷ = α + β_u + β_i + γ_u·γ_i` (course week 4)
Implemented in PyTorch with minibatch Adam. L2 is applied only to the rows touched by each batch (the usual SGD
equivalent of the course objective). We grid-search K × λ on `valid` and keep the epoch with the best validation MSE.

**Overfitting diagnosis and fix.** Our first grid used λ ≤ 1e-3. The penalty is `λ · mean ‖row‖²` next to an MSE of about 4, so it was
roughly 1/4000 of the loss and changing λ barely moved validation MSE. MF memorized the training set: validation MSE was best at
epoch 2 (4.24) and climbed to 6.1 by epoch 12 while train MSE fell to 1.2. We widened the grid to λ ∈ {1e-3, 0.03, 0.1, 0.15, 0.2}
and K ∈ {32, 64, 128}. The best setting (K=128, λ=0.15) reaches valid MSE 4.138 and stays flat after its best epoch
(epoch 12: 4.142), and λ ≥ 0.2 starts to underfit. The stacked models below are rebuilt on these better MF scores.
""")
code(r"""show_code("steam/regression_cf.py", "MF", "_train_torch")""")
md(r"""
### 3.3 Factorization machine (Rendle 2010): adding metadata
An FM represents every feature (user id, game id, each genre, top tags, price/year/metascore buckets, free-to-play,
early access, the user's library-size bucket) with a weight and a latent vector, and models **all pairwise interactions**
in O(kn) with the identity `½[(Σ v x)² − Σ (v x)²]`. With only user and game ids it reduces exactly to the latent
factor model, so it is a strict generalization. For never-seen games the id feature is masked and the prediction
comes from metadata alone.

The FM had the same regularization-scale problem as MF (λ = 1e-4 was negligible). Its penalty sums over the ~20 active
feature vectors of each row, not two. Our first widened grid still put the best λ at its edge, and the FM training
curve kept overfitting after epoch 4, so we extended the search to λ ∈ {1e-4, 0.01, 0.03, 0.1, 0.2} and K ∈ {32, 64, 128}.
""")
code(r"""show_code("steam/regression_cf.py", "FM")""")
code(r"""
t = pd.DataFrame(report.load("regression_cf")["fm"]).sort_values("valid_mse")
display(Markdown(f"**FM validation grid** (best: K={t.iloc[0].K:.0f}, λ={t.iloc[0]['lambda']:g}, valid MSE {t.iloc[0].valid_mse:.4f}; "
                 f"weakest-λ setting from our first version: {t[t['lambda'] == 1e-4].valid_mse.min():.4f})"))
t.round(4)
""")
md(r"""
### 3.4 Stacked LightGBM and 3.5 the hurdle model
Gradient boosting combines **leak-free engineered features** (user/game play rates and mean log-hours from `train`,
user × genre affinity, the share of the user's library in the game's genres, price, year, metascore, ...) with the
**outputs of 3.1–3.3**. It is trained on `valid` with 5-fold cross-fitting, so its training targets were never seen by the base models.

The **hurdle model** splits the problem in two. Since y = 0 whenever a game is unplayed,
`E[y] = P(played) · E[y | played]` is the MSE-optimal prediction. Stage 1 is exactly the Task 4 classifier.
""")
code(r"""report.fig_hurdle(); plt.show()""")
code(r"""show_code("steam/regression_stack.py", "crossfit_lgb", "run")""")
md(r"""
### 3.6 Recommendation models (Task 5)
- **Most-popular / Jaccard item-kNN**: course baselines; Jaccard over sets of players, summed over the user's games.
- **BPR-MF** (Rendle 2009): pairwise loss, played > not played.
- **Implicit ALS** (Hu, Koren & Volinsky 2008): *played* is a preference with confidence `1 + α·c`. We compare
  `c = 1` (**binary**) with `c = log(1 + minutes)` (**playtime-weighted**), which uses *how much*, not just *whether*.
- **item2vec**: skip-gram over each user's games sorted by playtime (as in HW4); user = playtime-weighted mean vector.
- **Content**: tf-idf over tags, genres, specs and developer; the user profile is a playtime-weighted mean.
- **Cold start**: a ridge map from content to ALS item factors, learned on warm games and applied to unseen ones.
- **Hybrid**: ALS + content, blended on **per-user z-scores** (rank-normalizing, as in our first version, threw away score gaps
  and made the hybrid *worse* than ALS); the content weight is tuned on `valid`.
- **Learning-to-rank reranker (LightGBM LambdaRank; Burges 2010)**: each candidate gets every base model's score, its per-user
  z-score, and game metadata, and LambdaRank learns how to combine them by directly optimizing NDCG. It is trained on the
  `valid` protocol and evaluated on `test`/`hard`, which hold different held-out games; the base models only saw `train`.
- **Task 4 → Task 5:** the hard protocol (played vs owned-but-unplayed) *is* the "will they play it?" question asked per user,
  so we also rank it with the LightGBM played-classifier from Section 3.5.

Hyperparameters tuned on `valid`: ALS confidence α, factors and L2; BPR factors, learning rate, L2 and epochs; hybrid weight.
""")
code(r"""show_code("steam/ranking.py", "fit_als", "fit_item2vec", "mapped_factor_scorer", "zscore_rows")""")
code(r"""show_code("steam/ranking_ltr.py", "candidate_features", "run")""")
md(r"""### What the models learn""")
code(r"""report.fig_training_curves(); plt.show()""")
code(r"""report.fig_tsne(); plt.show()""")
code(r"""report.fig_importance(); plt.show()""")
md(r"""
### Trade-offs between approaches
| Model | Strengths | Weaknesses | Cost here |
|---|---|---|---|
| Bias model | interpretable, closed form, strong baseline | no personalization beyond offsets | seconds |
| Latent factors | personal taste; embeddings reusable for ranking | cannot score unseen games; needs tuning | ~1 min per config |
| Factorization machine | adds metadata, handles cold start, generalizes MF | more hyperparameters; sparse-feature engineering | ~2 min per config |
| LightGBM stack | strongest tabular learner, feature importance | needs careful out-of-fold protocol to avoid leakage | ~1 min |
| Hurdle | matches the zero-inflated target; reuses the Task 4 classifier | two models to maintain | ~2 min |
| ALS (implicit) | fast, uses playtime as confidence | squared loss on 0/1 targets; popularity-leaning | ~10 s per config |
| BPR | optimizes ranking directly | slow to converge; sensitive to learning rate | ~1 min |
| Content / mapped factors | only option for unseen games | weaker on warm games | seconds |
| LambdaRank reranker | combines complementary signals; optimizes NDCG directly | needs every base model at serving time; needs its own held-out data | ~2 min |
""")

# =============================================================================
md(r"""
# 4. Evaluation

The evaluation code is one shared module, so every model is scored the same way. Ranking ties are broken
**pessimistically**, so a constant scorer cannot look good. Error bars for the factor models come from 3 random seeds.
""")
code(r"""show_code("steam/metrics.py", "ranking_report", "ber", "beyond_accuracy")""")
md(r"""## 4.1 Main task: hours played (regression)""")
code(r"""
reg = report.table_regression("test")
display(report.style_table(reg, best_low=["MSE", "MAE"], fmt="{:.4f}"))
b = reg.MSE.idxmin()
display(Markdown(f"**Best: {b}**, test MSE **{reg.loc[b,'MSE']:.3f}** vs **{reg.loc['Global mean','MSE']:.3f}** "
                 f"for the global mean (**{reg.loc[b,'vs global mean']:.0%} lower**) and "
                 f"{reg.loc['Bias model','MSE']:.3f} for the course bias model."))
""")
code(r"""report.fig_mse_bars("test"); plt.show()""")
md(r"""
**Are the gaps real?** A paired bootstrap over test pairs (500 resamples) gives a 95% interval for the difference in MSE
between consecutive models on the ladder. With ~500k test pairs even small gains are measurable, so the question becomes
whether they are *large enough to matter*.
""")
code(r"""report.bootstrap_regression().round(4)""")
code(r"""report.fig_tuning_heatmap(); plt.show()""")
code(r"""
from steam.regression_stack import breakdown
bd = breakdown("test")
report.fig_breakdown(bd); plt.show()
""")
md(r"""### Cold start: games never seen in training""")
code(r"""
cold = report.table_regression("cold")
display(report.style_table(cold, best_low=["MSE", "MAE"]))
""")
md(r"""
Collaborative models collapse on unseen games: the stacked models were trained where game statistics always exist.
The **metadata-only LightGBM** is the only regression model that clearly beats the user-mean baseline here. Whether a game
is in the training data is known at prediction time, so the deployable system **routes never-seen games to the
metadata-only model** and everything else to the ensemble:
""")
code(r"""
c = report.table_regression("cold"); t = report.table_regression("test")
display(Markdown(f"Routed system: test MSE **{t.loc['Ensemble','MSE']:.3f}** (warm games, ensemble) and cold MSE "
                 f"**{c.loc['LightGBM (metadata only)','MSE']:.3f}** (vs {c.loc['Ensemble','MSE']:.3f} if the ensemble were used on cold games, "
                 f"and {c.loc['User mean','MSE']:.3f} for the user-mean baseline)."))
""")
md(r"""## 4.2 Will they play it? (classification)""")
code(r"""
for t in ["played", "active"]:
    display(Markdown(f"**Target: `{t}`**"))
    display(report.style_table(report.table_classification(t), best_high=["AUC", "Accuracy"], best_low=["BER"]))
""")
code(r"""report.fig_roc(); plt.show()""")
md(r"""## 4.3 Recommending games they will play (ranking)""")
code(r"""
for p, title in [("test", "Standard protocol (1 played vs 99 not-owned)"),
                 ("hard", "Hard protocol (1 played vs 9 owned-but-unplayed)"),
                 ("cold", "Cold-start protocol (never-seen games)")]:
    display(Markdown(f"**{title}**"))
    display(report.style_table(report.table_ranking(p), best_high=["AUC", "HR@1", "HR@3", "HR@10", "NDCG@1", "NDCG@3", "NDCG@10"]))
""")
md(r"""
**Reading the ranking tables.**
- *Standard protocol:* popularity and Jaccard are hard to beat because the held-out game is usually a hit. Tuned ALS is the
  best single model on NDCG; the **LambdaRank reranker** beats every single model by learning *when* to trust kNN, ALS
  or popularity (its most-used inputs are the per-user z-scores of Jaccard, ALS and popularity).
- *Hard protocol:* every recommender drops sharply, because popularity can no longer separate the candidates (they are all owned).
  The **Task 4 played-classifier** is by far the best ranker here: it was trained on exactly this question and uses
  the user's own play rate, genre affinity and the CF scores.
- *Cold start:* only content-aware models work; mapping content to ALS factors and blending with tf-idf similarity is best.
- *BPR* trails ALS even after tuning learning rate, L2 and epochs (`implicit`'s BPR samples negatives uniformly, which
  mostly teaches it popularity); with sampled metrics it is the weakest collaborative model.
""")
code(r"""report.fig_ranking(); plt.show()""")
code(r"""
R = report.load("ranking")
pd.DataFrame({m: {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in d.items()} for m, d in R["seeds"].items()}).T
""")
md(r"""
**Is the reranker's gain significant, and what does it use?** Paired bootstrap over test users (1,000 resamples) of the
per-user NDCG@10 difference, and the reranker's feature importance (share of total split gain).
""")
code(r"""
display(report.table_ltr_significance().round(4))
display(report.ltr_importance().round(3))
""")
md(r"""## 4.4 Ablations: what each component is worth""")
code(r"""
display(report.ablations().round(4))
report.fig_ablation(); plt.show()
""")
md(r"""## 4.5 Popularity bias and diversity""")
code(r"""
display(pd.DataFrame(report.load("ranking")["beyond"]).T.round(3))
report.fig_popularity_bias(); plt.show()
""")
md(r"""
Accuracy on sampled metrics rewards recommending hits, since the held-out positive is usually popular. The table shows the
price: the more accurate models concentrate recommendations on the top 1% of games, while content-based models spread
exposure across the catalog. For a store this is a real trade-off (ethics lecture): long-tail and indie developers get
little exposure from accuracy-optimized recommenders.
""")
md(r"""## 4.6 Error analysis""")
code(r"""fig, worst = report.fig_errors(); plt.show(); worst""")
md(r"""
The largest errors are pairs with huge actual playtime that the model predicted as barely played, or the reverse.
Typical causes are idle or AFK games left running, free-to-play games, and users with enormous libraries whose behavior
is idiosyncratic. Predictions shrink toward the middle for the zero-playtime mass, which is the expected behavior of
an MSE-optimal model under uncertainty.
""")
md(r"""## 4.7 A concrete example""")
code(r"""
lib, recs = report.user_card()
display(Markdown("**This user's most-played games**")); display(lib)
display(Markdown("**Top-5 recommendations (playtime-weighted ALS), with an explanation**")); display(recs)
""")

# =============================================================================
md(r"""
# 5. Related work
""")
code(r"""report.table_related_work()""")
md(r"""
**How this dataset has been used.** The Steam crawl was introduced for sequential recommendation (SASRec; Kang & McAuley 2018),
for modeling chains of implicit signals such as purchase → play → review (Wan & McAuley 2018), and for bundle
recommendation (Pathak et al. 2017). Most prior work treats *ownership or interaction* as the positive signal.
Our framing makes the gap between owning and playing the object of study.

**How our results compare.**
- As in Koren et al. (2009), **biases capture a large share of the gain**, and latent factors add a further improvement.
- Hu et al. (2008) argue that **interaction strength should set confidence**. In our data, playtime-weighted ALS beats binary ALS
  on the *hard* protocol (choosing among owned games), but not on the standard protocol. Playtime helps decide which owned game
  gets played; it does not help predict which unowned game gets picked up.
- Ordering a library by playtime (item2vec, as in HW4) gives the best *single-signal recommender* on the hard protocol:
  embeddings that encode what a user plays *most* transfer to what they will play at all. But a supervised model trained
  on the actual question (our Task 4 classifier) beats every recommender there by a wide margin.
- As in the learning-to-rank literature (Burges 2010) and in Netflix Prize blends, **combining complementary rankers**
  beats any single one: the LambdaRank reranker improves NDCG@10 over the best single model by a statistically clear margin.
- Unlike SASRec's results, where most-popular is a weak baseline, **popularity is very strong** on the standard sampled
  protocol here. Libraries overlap heavily, and sampled metrics are known to favor popular items (Krichene & Rendle 2020).
  Our **hard protocol** removes this advantage and shows where personalization really matters.
""")

# =============================================================================
md(r"""
# Takeaways and limitations
""")
code(r"""
reg = report.table_regression("test"); rk = report.load("ranking")["results"]
best = reg.MSE.idxmin()
hb = max(rk["hard"], key=lambda m: rk["hard"][m]["AUC"])
ts = max(rk["test"], key=lambda m: rk["test"][m]["NDCG@10"])
single = max([m for m in rk["test"] if "LTR" not in m], key=lambda m: rk["test"][m]["NDCG@10"])
display(Markdown(f'''
1. **Owning ≠ playing.** {eda.glance().set_index("Statistic").loc["Owned games never played","Value"]} of owned games are never played,
   and engagement depends on price, genre and the user's own history.
2. **Playtime is predictable.** {best} reaches test MSE {reg.loc[best,"MSE"]:.3f} vs {reg.loc["Global mean","MSE"]:.3f} for the global mean
   ({1 - reg.loc[best,"MSE"] / reg.loc["Bias model","MSE"]:.0%} below the course bias model). Collaborative signal is by far
   the most valuable component; implicit-feedback affinities (ALS, item2vec) add to the explicit MF; the hurdle split adds a small further gain.
3. **Recommend for play, not purchase.** Popularity looks excellent on the standard protocol (AUC {rk["test"]["Most popular"]["AUC"]:.3f})
   but drops to {rk["hard"]["Most popular"]["AUC"]:.3f} when asked which *owned* games get played; {hb} reaches {rk["hard"][hb]["AUC"]:.3f}.
4. **Combine, then specialize.** {ts} reaches standard NDCG@10 {rk["test"][ts]["NDCG@10"]:.3f} vs {rk["test"][single]["NDCG@10"]:.3f}
   for the best single model ({single}); for owned games, the Task 4 classifier is the right tool.
5. **Regularization scale matters.** Our first MF/FM grids used λ values ~1/4000 of the loss; fixing the scale removed MF's
   overfitting and was the largest single gain for the main task.
'''))
""")
md(r"""
**Limitations / next steps:** a single split seed for the regression models (bootstrap intervals cover test-set noise but
not training randomness; seed error bars exist only for the ranking factor models); the regression is limited to owned pairs;
there is no temporal split (the crawl is a snapshot, so a sequential model like SASRec would need review timestamps); the
cold-start FM was not trained with id-dropout; sampled ranking metrics favor popular items (Krichene & Rendle 2020), and we did
not measure the reranker's catalog coverage; BPR remains weak with uniform negative sampling.
""")

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NB.parent.mkdir(exist_ok=True)
nbf.write(nb, NB)
print(f"wrote {NB} ({len(cells)} cells)")

if "--no-exec" not in sys.argv:
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=1800", "--ExecutePreprocessor.startup_timeout=300", str(NB)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", "--output", "workbook",
                    "--output-dir", str(ROOT), str(NB)], check=True, cwd=ROOT)
    html = (ROOT / "workbook.html").read_text()
    print("workbook.html starts with <!DOCTYPE html>:", html.lstrip().startswith("<!DOCTYPE html>"))
