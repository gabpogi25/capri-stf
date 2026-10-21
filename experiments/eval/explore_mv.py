"""
experiments/eval/explore_mv.py

EXPLORATION ONLY (docs/prereg/evaluation_plan.md, section "Exploration before
registration"). Seeds 96,000,000 + i, outside every confirmatory range. Fixes
the ACAT noise scale m0 from rigid windows at sigma = 0.5 px and checks run time
and the behaviour of each statistic on a few doses. Its numbers are motivation,
not evidence, and are never quoted as results.

Output: experiments/results/eval_exploration.csv
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")       # one BLAS thread per worker process

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import save_results  # noqa: E402
from synthetic_data.injection import make_window  # noqa: E402
import mv_window as mw  # noqa: E402

SEED = 96_000_000
CELLS = [("rigid", 0.0, 40)] + [(f, d, 15) for f in ("global_projective", "jitter", "drift", "warp") for d in (2.0, 8.0)] + [("swap", 0.1, 15)]


def run(task):
    family, dose, index, seed = task
    rng = np.random.default_rng(seed)
    window = make_window(rng, family, dose, 0.5)
    start = time.time()
    row = mw.window_statistics(window["tracks"], 0.5, seed)
    row.pop("_rec", None)
    return {"Family": family, "Dose": dose, "Index": index, "Seed": seed, "Induced_px": window["induced_px"],
            "Seconds": time.time() - start, **row}


def main() -> None:
    tasks, k = [], 0
    for family, dose, count in CELLS:
        for i in range(count):
            tasks.append((family, dose, i, SEED + k))
            k += 1
    with ProcessPoolExecutor(4) as pool:
        df = pd.DataFrame(list(pool.map(run, tasks)))
    null = df[df.Family == "rigid"]
    m0 = float(np.median(null["acat_scale"]) / 0.6745)
    print(f"m0 (ACAT noise scale) = {m0:.3e}; median seconds per window {df.Seconds.median():.2f}")
    stats = ["pencil_q10", "pencil_q50", "pencil_q90", "gauge_q90", "registration_rms", "f_ratio", "sampson_mean",
             "magsac_outlier", "deltaj_med"]
    for (family, dose), group in df.groupby(["Family", "Dose"]):
        print(family, dose, {s: round(float(np.mean(group[s] > null[s].quantile(0.95))), 2) for s in stats})
    df["ACAT_m0"] = m0
    save_results(df, "eval_exploration.csv", SEED, "evaluation_plan exploration (not evidence)")


if __name__ == "__main__":
    main()
