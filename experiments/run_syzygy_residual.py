"""
experiments/run_syzygy_residual.py

Plucker residual distribution across regimes on the independent-estimation model.

Why two measurements
--------------------
Brackets recomputed from one coherent point set give rho = 0 by Corollary 2.6
(docs/MATHEMATICS.md), and averaging the brackets of two configurations has no
physical meaning. Neither would measure the method.

Here every regime yields two independent measurements of its six points with
the same Gaussian noise (sigma = 1e-3 in the unit chart). Each of the 15
brackets is estimated from one measurement chosen at random. Under this model:
  * Generic / Near-Degenerate / Sparse-View: rho reflects measurement noise only;
  * Mismatched: measurement B has some points replaced; rho can respond only
    when the tuple contains a replaced point (column Contains_Corrupted);
  * Non-Rigid: measurement B is the deformed scene (displacement <= 0.05).

Three residuals are reported per trial: rho (Sigma_1..Sigma_3), rho_full
(all 15 relations; not blind at b*, Proposition 6.2) and rho_rel (scale-free).

Parameters: 1,000 trials per regime/sub-condition, 12 points per scene,
Mismatched swept over f in {0.05, 0.10, 0.20}. Seed 42005.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_experiment_seed, make_rng, save_results, random_six_tuple,
    regime_measurements, MEASUREMENT_NOISE_SIGMA,
)
from synthetic_data.generator import estimate_brackets_independently
from algorithms.syzygy.residual import (
    compute_syzygy_residual,
    compute_full_plucker_residual,
    compute_relative_plucker_residual,
)

EXPERIMENT_ID = 5
SEED = get_experiment_seed(EXPERIMENT_ID)  # 42005
N_TRIALS = 1000
N_POINTS = 12
CONDITIONS = [
    ("Generic", "none", {}),
    ("Near-Degenerate", "none", {}),
    ("Mismatched", "f=0.05", {"mismatch_fraction": 0.05}),
    ("Mismatched", "f=0.10", {"mismatch_fraction": 0.10}),
    ("Mismatched", "f=0.20", {"mismatch_fraction": 0.20}),
    ("Non-Rigid", "delta=0.05", {"deformation_bound": 0.05}),
    ("Sparse-View", "n=8", {}),
]


def run_experiment() -> pd.DataFrame:
    print("=" * 60)
    print("EXP 5: PLUCKER RESIDUAL DISTRIBUTION (independent estimation)")
    print("=" * 60)
    rows = []
    for condition_index, (regime, sub_condition, kwargs) in enumerate(CONDITIONS):
        n_points = 8 if regime == "Sparse-View" else N_POINTS
        for trial in range(N_TRIALS):
            trial_rng = make_rng(SEED + condition_index * 100000 + trial)
            _, measurements, corrupted = regime_measurements(regime, n_points, trial_rng, **kwargs)
            t = random_six_tuple(n_points, trial_rng)
            brackets, sources = estimate_brackets_independently(measurements, t, trial_rng)
            rows.append({
                "Regime": regime,
                "Sub_Condition": sub_condition,
                "Trial": trial,
                "rho": compute_syzygy_residual(brackets),
                "rho_full": compute_full_plucker_residual(brackets),
                "rho_rel": compute_relative_plucker_residual(brackets),
                "Contains_Corrupted": bool(set(t) & set(corrupted)),
                "Mixed_Sources": len(set(sources)) > 1,
                "Noise_Sigma": MEASUREMENT_NOISE_SIGMA,
                "Seed": SEED,
            })
    df = pd.DataFrame(rows)

    generic_median = df.loc[df.Regime == "Generic", "rho"].median()
    print("\nMedian residuals by condition (ratio vs Generic):")
    for (regime, sub), group in df.groupby(["Regime", "Sub_Condition"], sort=False):
        median_rho = group["rho"].median()
        print(f"  {regime:16s} {sub:11s} rho={median_rho:.2e} "
              f"rho_full={group['rho_full'].median():.2e} "
              f"rho_rel={group['rho_rel'].median():.2e} "
              f"ratio={median_rho / generic_median:.2f}")
    mismatched = df[df.Regime == "Mismatched"]
    for contains, group in mismatched.groupby("Contains_Corrupted"):
        print(f"  Mismatched, tuple contains corrupted point = {contains}: "
              f"median rho = {group['rho'].median():.2e} (n = {len(group)})")

    save_results(df, "syzygy_residual.csv", SEED,
                 "syzygy-residual experiment: Plucker residual distribution under independent bracket estimation")
    return df


if __name__ == "__main__":
    run_experiment()
