# MGTA 461 Assignment 2: Owning ≠ Playing (Steam)

Predict what Steam users **actually play**: hours played (main), will-they-play (classification),
and recommending games they'll play (ranking). Full plan in `PLAN.md`; the spec PDF is in `docs/`.

**Deliverables (29 marks: 5 per rubric section × 5 + 4 for peer grading):**
`workbook.html` (notebook exported as HTML, must start with `<!DOCTYPE html>`) + `video_url.txt`
(one line: Drive "anyone with the link" or YouTube URL to a ~20 min mp4) · `responses.txt` peer-grading report about a week later.

## Quick start

```bash
uv sync                                   # Python env (pyproject.toml / uv.lock)
uv run python scripts/run_all.py          # run any pipeline stage whose outputs are missing (~25 min from scratch)
uv run python scripts/build_notebook.py   # build + execute notebooks/workbook.ipynb, export workbook.html
```

Raw data goes in `data/` (4 `.json.gz` files from the [McAuley lab page](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)).
`artifacts/` and `figures/` are generated, so don't edit them by hand.

Rerun one stage after changing it: `uv run python scripts/run_all.py ranking` (stage names below).

## Layout: one module per plan task

| Stage (`run_all.py` name) | File | Plan task | What it does | Output |
|---|---|---|---|---|
| `data` | `steam/data.py` | 1 | parse + clean the 4 raw files | `artifacts/*.parquet` |
| `splits` | `steam/splits.py` | 1 | **the one shared split** + ranking protocols + leakage checks | `train/valid/test/cold.parquet`, `rank_*.npz` |
| – | `steam/features.py` | 1/3 | leak-free features (train-only stats, genre affinity, metadata) | – |
| `regression_cf` | `steam/regression_cf.py` | 3 | **PyTorch**: means, bias model, MF, factorization machine | `reg_base_*.parquet` |
| `regression_stack` | `steam/regression_stack.py` | 3 | **LightGBM**: stacked model, metadata-only model, hurdle, ensemble | `results/regression.json` |
| `classify` | `steam/classify.py` | 4 | played / active classifiers (logistic, LightGBM) | `results/classification.json` |
| `ranking` | `steam/ranking.py` | 5 | popularity, Jaccard, BPR, ALS (binary vs playtime), item2vec, content, cold start | `results/ranking.json` |
| – | `steam/metrics.py` | 6 | every metric (MSE, AUC/BER, HR/NDCG, coverage/novelty) | – |
| – | `steam/eda.py` | 2 | EDA numbers + figures F3–F9 | `figures/` |
| – | `steam/report.py` | 6/9 | tables + figures F1–F2, F10–F25 | `figures/` |
| – | `steam/viz.py` | 9 | shared style; model colors (grey baseline / blue course / orange advanced) | – |
| – | `scripts/build_notebook.py` | 8 | **notebook narrative lives here**; builds + exports `workbook.html` | `workbook.html` |

### Rules that keep results comparable
1. **Never change the split** (`splits.py`, `SEED=42`) without telling everyone, because every number depends on it.
2. Base models fit on `train` only. Stacked models fit on `valid` with 5-fold cross-fitting. Tune on `valid`, report `test`.
3. Anything computed from playtime uses `train` only; ownership (library contents) is a legal input.
4. **Don't import torch and LightGBM in the same process** (macOS OpenMP crash). That's why stages are separate processes.
5. Edit notebook text in `scripts/build_notebook.py`, not in the `.ipynb`.

## V2 backlog: pick a workstream

Each workstream maps to one or two files, so people can work in parallel without conflicts.

**A. Regression (`regression_cf.py`, `regression_stack.py`)**
- **MF overfits after epoch 2** (see F12): λ is far too weak and the best grid point is at its edge (largest K and λ) → try λ ∈ {1e-2, 3e-2, 1e-1}, K ∈ {128, 256}, lower lr. Likely the biggest single gain
- FM: id-dropout during training (mask game id ~30%) so metadata carries signal → better cold start
- LightGBM: tune num_leaves / learning rate; add embedding features, review-derived features
- Multi-seed error bars (3 seeds) for every regression model
- Ensemble: non-negative blend; separate cold-start blend

**B. Classification + hurdle (`classify.py`)**
- Calibration plot for P(played); precision@K per user
- `active` task: recency features, class weighting

**C. Ranking (`ranking.py`)**
- Tune BPR (lr, epochs) and ALS regularization; more seeds
- Stretch: SASRec-style sequential model on review timestamps (torch, so run it as its own stage)
- Bundle scoring (Pathak 2017) as an extension
- Explanations (F24): item2vec vs ALS similarity

**D. EDA + extra tasks (`eda.py`, new module)**
- Review sentiment task (tf-idf + logistic, BER), helpfulness
- User clustering (collectors vs grinders vs casuals) → per-cluster model accuracy

**E. Story, slides, video (`build_notebook.py`, `report.py`)**
- Tighten narrative + takeaways; check every claim against the final numbers
- Slides from `figures/`; record ~20 min (storyboard in `PLAN.md`)
- Related work: verify each "how we compare" line against final results
