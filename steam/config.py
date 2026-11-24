"""Paths and global constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"      # cached parquet / npz / json results
FIG = ROOT / "figures"        # PNGs for slides
ART.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEED = 42
SEEDS = [42, 7, 2025]         # for error bars
N_NEG = 99                    # sampled negatives for leave-one-out ranking
