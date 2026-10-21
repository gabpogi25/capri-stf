"""
experiments/run_separation_stability.py

Cross-Regime Residual Separation Ratio Stability

Checks whether the Generic-vs-Mismatched residual separation is stable
across n and measurement-noise levels.

Model (coherent recomputation would give rho = 0 for both regimes by
Corollary 2.6): two measurements of each scene with Gaussian noise sigma in
the unit chart; for Mismatched, measurement B has 20% of the points replaced.
Each bracket is estimated from one of the two measurements.

Parameters:
  - n ∈ {6, 20, 50, 100}
  - Noise grid: 20 log-spaced levels
  - Seed: 42109
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    anti_circularity_check, bootstrap_ci, random_six_tuple,
    STANDARD_NOISE_GRID, RESULTS_DIR,
)
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_mismatched_configuration,
    measure_with_gaussian_noise,
    estimate_brackets_independently,
)


FINDING_ID = 9
SEED = get_finding_seed(FINDING_ID)  # 42109
N_TRIALS = 200
N_VALUES = [6, 20, 50, 100]


def run_experiment():
    print("=" * 60)
    print("CROSS-REGIME RESIDUAL SEPARATION STABILITY")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    for n in N_VALUES:
        print(f"\n  n = {n}")

        for noise_idx, sigma in enumerate(STANDARD_NOISE_GRID):
            generic_rhos = []
            mismatched_rhos = []

            for trial in range(N_TRIALS):
                trial_rng = make_rng(SEED + n * 10000 + noise_idx * 1000 + trial)

                t = random_six_tuple(n, trial_rng)

                # Generic: two noisy measurements of the same scene
                pts_gen = generate_generic_configuration(n, scale=1.0, rng=trial_rng)
                measurements = [measure_with_gaussian_noise(pts_gen, sigma, trial_rng),
                                measure_with_gaussian_noise(pts_gen, sigma, trial_rng)]
                brackets, _ = estimate_brackets_independently(measurements, t, trial_rng)
                rho_gen = compute_syzygy_residual(brackets)
                generic_rhos.append(rho_gen)

                # Mismatched: measurement B has replaced points
                pts_orig, pts_mis, _ = generate_mismatched_configuration(
                    n, mismatch_fraction=0.2, scale=1.0, rng=trial_rng
                )
                measurements = [measure_with_gaussian_noise(pts_orig, sigma, trial_rng),
                                measure_with_gaussian_noise(pts_mis, sigma, trial_rng)]
                brackets_mis, _ = estimate_brackets_independently(measurements, t, trial_rng)
                rho_mis = compute_syzygy_residual(brackets_mis)
                mismatched_rhos.append(rho_mis)

                results.append({
                    "N": n,
                    "Noise_Sigma": sigma,
                    "Trial": trial,
                    "Rho_Generic": rho_gen,
                    "Rho_Mismatched": rho_mis,
                    "Seed": SEED,
                })

            # Separation ratio for this (n, sigma)
            med_gen = np.median(generic_rhos)
            med_mis = np.median(mismatched_rhos)
            ratio = med_mis / max(med_gen, 1e-30)

            if noise_idx % 5 == 0:
                print(f"    \u03c3={sigma:.1e}: sep ratio = {ratio:.2e}")

    df = pd.DataFrame(results)

    # ---- Stability analysis ----
    print("\n--- Separation Ratio Stability ---")
    for n in N_VALUES:
        sub = df[df["N"] == n]
        ratios = []
        for sigma in STANDARD_NOISE_GRID:
            sub_sigma = sub[sub["Noise_Sigma"] == sigma]
            med_gen = np.median(sub_sigma["Rho_Generic"].values)
            med_mis = np.median(sub_sigma["Rho_Mismatched"].values)
            ratio = med_mis / max(med_gen, 1e-30)
            ratios.append(ratio)
        ratios = np.array(ratios)
        log_ratios = np.log10(ratios + 1e-30)
        print(f"  n={n}: mean log10(ratio) = {np.mean(log_ratios):.2f}, "
              f"std = {np.std(log_ratios):.2f}, "
              f"range = [{np.min(log_ratios):.2f}, {np.max(log_ratios):.2f}]")

    save_results(df, "separation_stability.csv", SEED,
                 "Separation ratio stability")


if __name__ == "__main__":
    run_experiment()
