"""LightGBM regularization search for the stacked and metadata-only models (VALID 5-fold OOF only).

    uv run python scripts/tune_lgb.py     # writes artifacts/results/lgb_tuning.json

Chosen setting is written into steam/regression_stack.py (LGB_REG). Test data is never touched here.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from steam import splits, features
from steam.regression_stack import crossfit_lgb, LGB_REG, FIT_GAP
rows = []
va = splits.load("valid"); base = pd.read_parquet("artifacts/reg_base_valid.parquet")
X = features.stack_features("valid", va, base); y = va.y.values
meta = [c for c in X.columns if c.startswith(features.META_PREFIXES)]
grid = {
 "V2 setting": dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, lambda_l2=1.0, feature_fraction=0.8),
 "reg_A": dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, lambda_l2=10.0, feature_fraction=0.7),
 "reg_B": dict(num_leaves=127, min_data_in_leaf=500, lambda_l2=20.0, feature_fraction=0.6, learning_rate=0.03),
 "reg_C": dict(num_leaves=255, min_data_in_leaf=1000, lambda_l2=10.0, feature_fraction=0.6, learning_rate=0.03, min_gain_to_split=0.01),
}
for cols, tag in [(list(X.columns), "full"), (meta, "meta")]:
    for k, g in grid.items():
        oof, _, _ = crossfit_lgb(X[cols], y, [], {**LGB_REG, **g}, name=f"{tag}:{k}")
        r = FIT_GAP[f"{tag}:{k}"]
        rows.append({"model": tag, "setting": k, **g, "oof_mse": r["oof_mse"], "train_mse": r["train_mse"], "trees": r["rounds"]})
        print(f"{tag:5s} {k:8s} OOF {r['oof_mse']:.4f}  train {r['train_mse']:.4f}  gap {r['oof_mse']-r['train_mse']:.3f}  trees {r['rounds']:.0f}", flush=True)
json.dump(rows, open("artifacts/results/lgb_tuning.json", "w"), indent=1)
