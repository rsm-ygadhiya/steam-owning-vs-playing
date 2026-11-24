"""Run every pipeline stage in order, each in its own process.

    uv run python scripts/run_all.py            # run stages whose outputs are missing
    uv run python scripts/run_all.py --force    # rerun everything
    uv run python scripts/run_all.py ranking    # (re)run one stage by name

Stages run as separate processes because torch (regression_cf) and LightGBM
(regression_stack, classify) bundle clashing OpenMP runtimes on macOS.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

STAGES = [  # (name, module, output that proves it ran)
    ("data", "steam.data", "interactions.parquet"),
    ("splits", "steam.splits", "train.parquet"),
    ("regression_cf", "steam.regression_cf", "reg_base_test.parquet"),
    ("regression_stack", "steam.regression_stack", "results/regression.json"),
    ("classify", "steam.classify", "results/classification.json"),
    ("ranking", "steam.ranking", "results/ranking.json"),
]

CMD = {
    "steam.data": "from steam import data; print(data.build(force=True))",
    "steam.splits": "from steam import splits; splits.build(force=True); print(splits.stats())",
}


def run(name, module):
    t = time.time()
    code = CMD.get(module)
    cmd = [sys.executable, "-c", code] if code else [sys.executable, "-m", module]
    print(f"==> {name}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"<== {name} ({time.time() - t:.0f}s)", flush=True)


def main(argv):
    force = "--force" in argv
    only = [a for a in argv if not a.startswith("--")]
    for name, module, out in STAGES:
        if only and name not in only:
            continue
        if not only and not force and (ART / out).exists():
            print(f"--- {name}: cached ({out})")
            continue
        run(name, module)


if __name__ == "__main__":
    main(sys.argv[1:])
