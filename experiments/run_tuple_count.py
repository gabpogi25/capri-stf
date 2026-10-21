"""
experiments/run_tuple_count.py

Minimal Tuple Count k(n) Scaling

Hypothesis: The minimal accepted tuple count k tracks k = 3(n-3).

Parameters:
  - n ∈ standard point-count grid (10 to 100 step 5)
  - 1,000 trials per n value
  - Seed: 42108
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    anti_circularity_check, bootstrap_ci,
    fit_power_law_loglog, STANDARD_N_GRID, RESULTS_DIR,
)
from algorithms.selection.greedy import select_tuples_greedy
from synthetic_data.generator import generate_generic_configuration


FINDING_ID = 8
SEED = get_finding_seed(FINDING_ID)  # 42108
N_TRIALS = 1000


def run_experiment():
    print("=" * 60)
    print("MINIMAL TUPLE COUNT k(n) SCALING")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    for n in STANDARD_N_GRID:
        print(f"\n  n = {n}")
        k_values = []
        theoretical_k = 3 * (n - 3)

        for trial in range(N_TRIALS):
            trial_rng = make_rng(SEED + n * 10000 + trial)
            pts = generate_generic_configuration(n, scale=1.0, rng=trial_rng)

            # Route through C++ for blazing fast evaluation!
            try:
                import capri_cpp
                selected = capri_cpp.select_tuples_greedy(
                    points=pts,
                    max_tuples=max(500, theoretical_k * 2),
                    conditioning_threshold=1e6,
                    seed=(SEED + n * 10000 + trial) % 2147483647
                )
            except ImportError:
                print("WARNING: C++ module not found, falling back to pure Python.")
                selected = select_tuples_greedy(
                    pts,
                    conditioning_threshold=1e6,
                    max_tuples=max(500, theoretical_k * 2),
                    rng=trial_rng,
                )
            k_empirical = len(selected)
            k_values.append(k_empirical)

            results.append({
                "Point_Count_n": n,
                "Trial": trial,
                "Empirical_k": k_empirical,
                "Theoretical_k": theoretical_k,
                "Residual": k_empirical - theoretical_k,
                "Seed": SEED,
            })

        mean_k, ci_lo, ci_hi = bootstrap_ci(np.array(k_values), rng=make_rng(SEED + 999))
        mean_residual = np.mean(np.array(k_values) - theoretical_k)
        print(f"    Theoretical k = {theoretical_k}")
        print(f"    Empirical k = {mean_k:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]")
        print(f"    Mean residual (empirical - theoretical) = {mean_residual:.2f}")

    df = pd.DataFrame(results)

    # ---- Residual trend analysis ----
    print("\n--- Residual Trend Analysis ---")
    agg = df.groupby("Point_Count_n")["Residual"].agg(["mean", "std"]).reset_index()
    print(agg.to_string(index=False))

    # Fit residual vs n to check for systematic bias
    residual_means = agg["mean"].values
    n_values = agg["Point_Count_n"].values
    if np.std(residual_means) > 1e-10:
        coeffs = np.polyfit(n_values, residual_means, 1)
        print(f"\n  Linear fit of residual vs n: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.2f}")
        if abs(coeffs[0]) < 0.1:
            print(f"  Slope near zero -> k = 3(n-3) holds (residual is noise)")
        else:
            print(f"  Non-zero slope -> correction term may be needed")
    else:
        print(f"  Residual variance too low -- anti-circularity concern")

    save_results(df, "tuple_count.csv", SEED, "Minimal tuple count scaling")


if __name__ == "__main__":
    run_experiment()
