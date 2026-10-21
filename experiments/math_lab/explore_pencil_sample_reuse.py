"""
experiments/math_lab/explore_pencil_sample_reuse.py

EXPLORATORY (not confirmatory; plan item B3). Video windows with the same number of
points can share one sampled pencil set (cached by algorithms/consistency/pencil.py) instead of
drawing a fresh sample per window. This compares the two on the frame-free drift benchmark of
run_pencil_drift.py, with n = 20 points so that the full family (1,744,200 pencils) is computable
and both sampled statistics can be compared with the exact one:
  shared   one sample of 50,000 pencils for every window (seed 0, the library default)
  fresh    a new sample of 50,000 pencils per window (seed SEED + scene)
  exact    the full family
Reported: AUC of each with paired bootstrap CIs of shared - fresh and shared - exact; the
relative error of each sampled statistic against the exact one; time per window.
Seeds from the exploration band (90.4e6; unused before this script).

Run: python experiments/math_lab/explore_pencil_sample_reuse.py [--scenes N]
Output: pencil_sample_reuse.csv (per scene), pencil_sample_reuse_summary.csv.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from experiment_utils import save_results  # noqa: E402
import lab  # noqa: E402
from algorithms.consistency.pencil import pencil_drift_score, pencil_family_size  # noqa: E402
from synthetic_data.generator import generate_generic_configuration, perturb_within_ball  # noqa: E402

SEED = 90_400_000
N_POINTS = 20
K = 50_000
N_BOOT = 1000


def scene(rng) -> tuple[dict, int]:
    """lab.gen_drift with N_POINTS points."""
    points = generate_generic_configuration(N_POINTS, scale=1.0, rng=rng)
    delta = float(np.exp(rng.uniform(np.log(0.003), np.log(0.1))))
    label = int(rng.random() < 0.5)
    radius = delta * (rng.uniform(1.5, 5.0) if label else 1.0)
    A = lab.gauge(perturb_within_ball(points, radius, rng), rng)
    B = lab.gauge(perturb_within_ball(points, radius, rng), rng)
    return {"A": A, "B": B, "delta": delta}, label


def timed(fn) -> tuple[float, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, 1e3 * (time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", type=int, default=600)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    family = pencil_family_size(N_POINTS)
    rows = []
    for k in range(args.scenes):
        s, y = scene(rng)
        A, B, d = s["A"], s["B"], s["delta"]
        shared, t_shared = timed(lambda: pencil_drift_score(A, B, d, max_pencils=K, seed=0))
        fresh, t_fresh = timed(lambda: pencil_drift_score(A, B, d, max_pencils=K, seed=SEED + 1 + k))
        exact, t_exact = timed(lambda: pencil_drift_score(A, B, d, max_pencils=family))
        rows.append({"Scene": k, "Label": y, "Shared": shared, "Fresh": fresh, "Exact": exact,
                     "Ms_shared": t_shared, "Ms_fresh": t_fresh, "Ms_exact": t_exact})
    df = pd.DataFrame(rows)
    y = df["Label"].to_numpy()
    boot = np.random.default_rng(SEED + 999_999)
    idx = [boot.integers(0, len(df), len(df)) for _ in range(N_BOOT)]
    summary = []
    for name in ("Shared", "Fresh", "Exact"):
        s = df[name].to_numpy()
        row = {"Statistic": name, "AUC": lab.auc(y, s)}
        if name != "Shared":
            diff = [lab.auc(y[i], df["Shared"].to_numpy()[i]) - lab.auc(y[i], s[i]) for i in idx]
            row["dAUC_shared_minus_this_Lo"], row["dAUC_shared_minus_this_Hi"] = np.percentile(diff, [2.5, 97.5])
        if name != "Exact":
            rel = np.abs(s / df["Exact"].to_numpy() - 1)
            row["RelErr_vs_exact_median"], row["RelErr_vs_exact_max"] = float(np.median(rel)), float(rel.max())
        row["Median_ms_per_window"] = float(df[f"Ms_{name.lower()}"].iloc[1:].median())   # first call fills the cache
        summary.append(row)
    print(pd.DataFrame(summary).to_string(index=False))
    meta = f"EXPLORATORY B3: shared vs fresh pencil samples, n = {N_POINTS}, K = {K}, {len(df)} scenes"
    save_results(df, "pencil_sample_reuse.csv", SEED, meta)
    save_results(pd.DataFrame(summary), "pencil_sample_reuse_summary.csv", SEED, meta)


if __name__ == "__main__":
    main()
