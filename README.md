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
| `ranking` | `steam/ranking.py` | 5 | popularity, Jaccard, BPR, ALS (binary vs playtime), item2vec, content, z-score hybrid, cold start; dumps candidate scores | `results/ranking.json`, `rank_feats_*.npz` |
| `regression_stack` | `steam/regression_stack.py` | 3 | **LightGBM**: stacked model (+ ALS / item2vec affinities), metadata-only model, hurdle, ensemble | `results/regression.json` |
| `classify` | `steam/classify.py` | 4 | played / active classifiers (logistic, LightGBM) | `results/classification.json` |
| `ranking_ltr` | `steam/ranking_ltr.py` | 5 | **LightGBM LambdaRank** reranker over all recommenders; Task 4 classifier on the hard protocol; paired bootstrap | `results/ranking_ltr.json` |
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

## V2 changes (done)

- **MF / FM regularization scale fixed**: λ ≤ 1e-3 was ~1/4000 of the loss. MF valid MSE 4.244 → 4.138; FM grid widened the same way.
- **Stacking uses implicit-feedback affinities** (ALS dot, item2vec similarity) → ensemble test MSE 4.066 → ~3.92.
- **Ranking**: ALS L2 tuned; BPR lr/L2/epochs tuned (still weak); hybrid blends per-user z-scores (now beats ALS; was worse).
- **New LambdaRank reranker** (`ranking_ltr.py`): best on the standard protocol (NDCG@10 ≈ 0.78 vs 0.71 for the best single model),
  early-stopped on a pessimistic-tie NDCG (LightGBM's built-in NDCG treats ties optimistically and stopped at round 1).
- **Task 4 classifier ranks the hard protocol**: AUC ≈ 0.86 vs 0.76 for the best recommender.
- **Significance**: paired bootstrap CIs for the regression ladder and the reranker's gains.
- Stage order changed: `ranking` now runs before `regression_stack` (its factors are features).

## Remaining ideas

- FM id-dropout for cold start; multi-seed error bars for the regression models
- Reranker catalog coverage (beyond-accuracy) and a hard-protocol-specific reranker
- Review sentiment task; user clustering; SASRec on review timestamps; bundle scoring
- Slides + ~20 min video (storyboard in `PLAN.md`)
