# Assignment 2 Plan — Steam: Predicting What Gamers Play

**Due:** ~Tue Dec 1 (week 10; confirm on Gradescope)
**Deliverables:** `workbook.html` (notebook exported as HTML) + `video_url.txt` (~20 min mp4 on Drive, "anyone with the link") · peer grading `responses.txt` the week after
**Rubric:** 5 sections × 5 marks. A 5 means "above and beyond": thorough, an interesting or new approach. This plan aims for 5s.

---

## Dataset — Steam (Australian users, McAuley lab)

Source: [McAuley lab datasets page, Steam section](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data). Direct downloads: [users_items](https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz) (71 MB) · [user_reviews](https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz) (6.7 MB) · [steam_games](https://cseweb.ucsd.edu/~wckang/steam_games.json.gz) (2.7 MB) · [bundle_data](https://mcauleylab.ucsd.edu/public_datasets/data/steam/bundle_data.json.gz) (92 KB)

| File | What's in it | Size |
|---|---|---|
| `australian_users_items.json.gz` | Each user's library with `playtime_forever` and `playtime_2weeks` (minutes) | 88,310 users (16,806 empty), **5.15M user–game pairs**, 10,978 games; median 26 games/user; **36% of owned games never played** |
| `australian_user_reviews.json.gz` | Review text, `recommend`, posted date, helpful/funny votes | 59,305 reviews, 25,799 users, 3,682 games; 88% recommend = True |
| `steam_games.json.gz` | Genres, tags, specs, price, developer, publisher, release date, metascore, sentiment, early access | 32,135 games |
| `bundle_data.json.gz` | Bundles, their games, prices and discounts | 615 bundles |

Files are Python literals, so load them with `ast.literal_eval`.

---

## Core story

> **"Owning a game isn't the same as playing it."** Given a user's Steam library, predict (1) which games they will actually engage with and (2) how much. Then use that to recommend games they would play, not just buy.

The deck is built around one main task plus two connected tasks, all sharing one data pipeline and one evaluation framework.

---

## Work plan

### Task 1 — Data pipeline and shared split
- Load and join the four files; map user/game IDs to integers; drop empty libraries; join metadata (genres, tags, price, release year, metascore).
- Targets: `log(1 + playtime_forever)`, `played = playtime > 0`, `active = playtime_2weeks > 0`.
- **Splits** (fixed seed, saved to disk so every result uses the same split):
  - Random 80/10/10 over user–game pairs, for playtime regression.
  - **Leave-one-out per user**, for ranking (hold out one played game per user, plus 99 sampled negatives).
  - **Cold-start split:** 10% of games held out entirely, to test metadata-based models.
- Sanity checks: no leakage between splits, and the number of users/games present in each.
- **Leakage specs (decide now, before modeling):**
  - **Engineered features in 3.4:** user/game biases, latent dot products and "user's average playtime in this genre" must be computed out-of-fold or on the training split only, never from the row being predicted. Otherwise the target leaks into its own features.
  - **Task 4 (`active`):** if the target uses `playtime_2weeks`, do not give the model `playtime_forever` as a feature.
  - **Task 5 negatives:** define the 99 sampled negatives explicitly (unplayed owned games, non-owned games, or a fixed mix with stated proportions), and use the same definition for every model so the metric means the same thing.

### Task 2 — Exploratory analysis (rubric section 2)
- Playtime distribution (raw vs. log), share of games never played, games per user (long tail), the "pile of shame" (owned but unplayed) by genre/price.
- Popularity concentration: what share of total playtime the top 1% of games get (Lorenz curve / Gini).
- Playtime vs. price, genre, free-to-play, release year, metascore, early access.
- Reviews: recommend rate, review length vs. recommend, how reviewers' playtime compares with non-reviewers'.
- Tags/genres co-occurrence heat map; t-SNE of game embeddings (after Task 5) as a visual.

### Task 3 — Main task: playtime prediction (regression)
Predict `log(1 + playtime_forever)` for a (user, game) pair in the user's library. **Metric:** MSE (plus MAE), with breakdowns by user activity and game popularity.

| # | Model | Why |
|---|---|---|
| 3.0 | Global / user / game mean | Trivial baselines |
| 3.1 | Bias model `α + β_u + β_i`, λ tuned | Course week 4 |
| 3.2 | Latent factor model (tune K, λ; SGD or ALS) | Course week 4 |
| 3.3 | **Factorization Machine** with user, game, genres, tags, price, release year, user library size | Combines collaborative filtering with metadata and handles cold start. Main "above and beyond" model |
| 3.4 | **Gradient boosting** (LightGBM) on engineered features: user/game biases from 3.1, latent dot product from 3.2, game metadata, user's average playtime in this genre, library size, share of user's library in genre | Strong tabular comparison; feature importance makes a good slide |
| 3.5 | **Hurdle (two-stage) model:** P(played) × E[log time ∣ played] | Handles the 36% zeros, which a plain MSE model handles poorly |
| 3.6 | Ensemble (stack the best models) | Final-number squeeze |

### Task 4 — Connected task A: "Will they play it?" (classification)
Predict `played` (and separately `active` in the last 2 weeks). **Metrics:** AUC, BER (classes are 64/36), precision@K.
- Logistic regression on metadata + user features (course week 2) → latent-factor scores as features → LightGBM.
- This classifier is stage 1 of the hurdle model in 3.5, so one piece of work serves two tasks.

### Task 5 — Connected task B: top-N game recommendation (ranking)
Given a user's library, rank the games they will **actually play** (positive = played > 0; unplayed owned games are *hard negatives*, which is the interesting twist).
**Metrics:** AUC, HitRate@10, NDCG@10 with leave-one-out + 99 sampled negatives; also catalog coverage and novelty.

| # | Model | Why |
|---|---|---|
| 5.0 | Most-popular; Jaccard item-to-item kNN | Course baselines (week 4, HW2) |
| 5.1 | **BPR-MF** | Course implicit-feedback model (A1) |
| 5.2 | **Playtime-weighted implicit ALS** (Hu, Koren & Volinsky 2008: confidence = 1 + α·log playtime) | Uses *how much*, not just *whether* |
| 5.3 | **item2vec** on libraries ordered by playtime (as in HW4) + user = average of game vectors | Course week 5 |
| 5.4 | **Content-boosted / cold-start:** tag + genre embeddings (tf-idf over tags) blended with 5.2 | Evaluated on the cold-start split |
| 5.5 | *(stretch)* Sequential model on review timestamps (SASRec-style, Kang & McAuley 2018) | Links directly to the dataset's paper |

### Task 6 — Evaluation framework (rubric section 4)
- One shared evaluation module: MSE/MAE, AUC/BER, HR@K/NDCG@K, coverage, novelty.
- Results table across all models × all tasks; validation curves for K and λ; error bars over 3 seeds.
- **Breakdowns:** cold vs. warm users and games; popular vs. long-tail games; by genre.
- **Ablations:** remove metadata, remove playtime weighting, remove the hurdle stage, and measure the drop for each.
- **Popularity-bias and diversity analysis:** do our recommenders only push top games? Compare long-tail exposure across models. Ties to the course's ethics lecture.
- Error analysis: the worst predictions (e.g. idle games left running, free-to-play games, games bundled but never opened).

### Task 7 — Related work (rubric section 5)
- **Dataset papers:** Kang & McAuley 2018 (SASRec, ICDM); Wan & McAuley 2018 (monotonic behavior chains, RecSys); Pathak, Gupta & McAuley 2017 (Steam bundles, SIGIR).
- **Methods:** Koren, Bell & Volinsky 2009 (matrix factorization); Rendle 2009 (BPR); Rendle 2010 (Factorization Machines); Hu, Koren & Volinsky 2008 (implicit ALS); Barkan & Koenigstein 2016 (item2vec).
- Compare our numbers against the trends reported in those papers (e.g. BPR vs. popularity gap, the value of sequence information).

### Task 8 — Notebook, slides, video, submission (visuals planned in Task 9)
- One clean notebook, organized by the 5 rubric sections, with markdown explaining every step; precomputed results so it renders fully as HTML.
- Slides + code walkthrough, ~20 minutes (graders may watch at 1.5× if it runs long).
- Export `workbook.html` (must start with `<!DOCTYPE html>`), upload the mp4 to Drive, write `video_url.txt`, and run the autograder checks.

### Task 9 — Visuals and video storyboard

Every figure is made **in the notebook** (so it appears in `workbook.html` and the code matches the video). It's also saved to `figures/` as a PNG for the slides. Use one consistent style: same palette, model colors fixed across all charts (e.g. baselines grey, course models blue, advanced models orange), readable axis labels, and a one-line takeaway as each chart's title ("36% of owned games are never played", not "Playtime histogram").

#### Figure list (by rubric section)

| # | Figure / table | Type | The point it makes |
|---|---|---|---|
| **§1 Task** | | | |
| F1 | Pipeline diagram: library → played? → how long? → recommend | Diagram | The whole project in one picture (opening slide) |
| F2 | Task × metric × baseline table | Table | What we predict, how we score it, what we must beat |
| **§2 Data / EDA** | | | |
| F3 | Dataset at a glance (users, games, pairs, reviews, sparsity) | Stat tiles | Scale and sparsity |
| F4 | Playtime histogram, raw vs. log(1+x), with the zero spike marked | Histogram pair | Why we log-transform; the 36% zeros motivate the hurdle model |
| F5 | Games per user (log-log) + playtime share Lorenz curve with Gini | Line charts | Long tail: a few games get most of the playtime |
| F6 | "Pile of shame": % of owned games unplayed, by genre and by price band | Bar chart | Owning ≠ playing, and it varies by genre/price |
| F7 | Median playtime by genre / free-to-play / release year | Box or bar | Metadata carries signal, so it's worth adding features |
| F8 | Top 15 games by owners vs. by total hours | Paired bar | Popular to own ≠ popular to play |
| F9 | Review recommend rate + reviewer vs. non-reviewer playtime | Bar | Review data adds context |
| **§3 Modeling** | | | |
| F10 | Model ladder diagram (mean → biases → MF → FM → LightGBM → hurdle → ensemble) | Diagram | How each model builds on the last |
| F11 | Hurdle model schematic: P(played) × E[hours ∣ played] | Diagram | Explains the two-stage idea visually |
| F12 | Training curves (train vs. valid MSE over epochs) for MF/FM | Line | Convergence and overfitting control |
| F13 | t-SNE of game embeddings (item2vec / MF), colored by genre | Scatter | The model learns genres without being told |
| F14 | LightGBM feature importance (top 15) | Horizontal bar | What drives playtime |
| **§4 Evaluation** | | | |
| F15 | Main results table: all models × all metrics (best in bold, ±std over 3 seeds) | Table | The headline numbers |
| F16 | Test MSE by model, ordered, with baseline line | Bar with error bars | Each step up the ladder helps (or not) |
| F17 | Validation MSE vs. K and λ (heat map) | Heat map | How we tuned; no test-set peeking |
| F18 | MSE by user activity and game popularity bins (cold → warm) | Grouped bar | Where models win and fail; cold start |
| F19 | Ranking: HR@10 / NDCG@10 per model + cold-start split | Grouped bar | Recommendation quality |
| F20 | ROC curves for "will they play?" | Line | Classification quality (AUC) |
| F21 | Ablation waterfall: full model → remove metadata / hurdle / playtime weight | Waterfall bar | The marginal value of each component |
| F22 | Popularity bias: share of recommendations from top-1% games vs. long tail, per model | Stacked bar | Accuracy vs. diversity trade-off (ethics tie-in) |
| F23 | Predicted vs. actual log playtime (hexbin) + 3 worst-error examples | Hexbin + table | Error analysis (idle games, free-to-play, bundles) |
| F24 | One user example: library, top-5 recommendations with "because you played X" | Table card | Concrete demo; memorable slide |
| **§5 Related work** | | | |
| F25 | Related-work table: paper, task, method, metric, how we compare | Table | Positions our work against prior results |

#### Video storyboard (~20 min; slides + code walkthrough)

| Time | Section | On screen | Figures |
|---|---|---|---|
| 0:00–1:00 | Hook | "Steam users own a median of 26 games, and over a third of owned games are never opened" | F3, F1 |
| 1:00–3:30 | §1 Task | Three tasks, metrics, baselines, validity checks | F1, F2 |
| 3:30–7:30 | §2 Data | Source/context, cleaning steps (brief notebook scroll), EDA story | F4–F9 |
| 7:30–12:30 | §3 Modeling | Model ladder; walk through the MF, FM and hurdle code; embeddings and feature importance | F10–F14 |
| 12:30–17:30 | §4 Evaluation | Results, tuning, cold start, ranking, ablations, bias, error analysis, user demo | F15–F24 |
| 17:30–19:30 | §5 Related work | How the dataset was used before; how our results compare | F25 |
| 19:30–20:00 | Wrap-up | Three takeaways + limitations | — |

Recording tips: 1080p screen capture, notebook zoomed to ~125%, slides 16:9, and keep each section under its time box (graders may switch to 1.5× if it runs long).

---

## More ideas (pick 1–2 if there's time; each could be its own slide)

1. **Review sentiment task:** predict `recommend` from review text (tf-idf + logistic regression → sentence embeddings), then test whether adding user/game biases helps. A text-mining tie-in (week 5). The 88/12 imbalance makes BER the right metric.
2. **Review helpfulness prediction:** predict the helpful-vote ratio from text length, playtime at review time, and sentiment.
3. **Bundle recommendation:** score each bundle for a user as the aggregate of predicted game scores, and compare with the bundle paper (Pathak 2017).
4. **Price sensitivity:** do users play cheap or discounted games less? Interact price with user features in the FM.
5. **"Pile of shame" profiler:** cluster users by owned-vs-played behavior (collectors vs. grinders vs. casuals) with k-means on library features, then show per-cluster model accuracy.
6. **Explainable recommendations:** "Because you played X for 200 hours", using item-to-item similarity from item2vec or ALS.
7. **Fairness/exposure:** compare recommendation exposure for indie vs. major-publisher games.
8. **Currently-active prediction:** use `playtime_2weeks` as a "what will they play this week" task, a proxy for a temporal model.

---

---

## Open decisions
- Confirm the scope: core (Tasks 1–9) + which ideas from "More ideas".
- Shared workspace: Google Colab vs. a git repo.
- Compute: 5.15M pairs is fine on a laptop for MF/ALS/LightGBM. Subsample users for the stretch sequential model.
