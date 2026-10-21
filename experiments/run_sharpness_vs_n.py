"""
experiments/run_sharpness_vs_n.py

Stratum Boundary Sharpness vs n

Parameters:
  - Repeat the boundary-shape κ-sweep at n ∈ {10, 20, 40, 80}
  - 500 trials per (n, κ) combination
  - Seed: 42107
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    generate_near_degenerate_config, pick_first_six_tuple, RESULTS_DIR,
)
from algorithms.conditioning.scores import compute_delta_min


FINDING_ID = 7
SEED = get_finding_seed(FINDING_ID)  # 42107
N_VALUES = [10, 20, 40, 80]
N_KAPPA_STEPS = 40
N_TRIALS = 500


def run_experiment():
    print("=" * 60)
    print("STRATUM BOUNDARY SHARPNESS VS N")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    kappa_grid = np.linspace(0, 0.999, N_KAPPA_STEPS)

    for n in N_VALUES:
        print(f"\n  n = {n}")
        for kappa_idx, kappa in enumerate(kappa_grid):
            delta_mins = []
            for trial in range(N_TRIALS):
                trial_rng = make_rng(SEED + n * 100000 + kappa_idx * 1000 + trial)
                pts = generate_near_degenerate_config(n, kappa=kappa, rng=trial_rng)
                t = pick_first_six_tuple(min(6, n))
                dm = compute_delta_min(pts, t)
                delta_mins.append(dm)

                results.append({
                    "N": n,
                    "Kappa": kappa,
                    "Trial": trial,
                    "Delta_Min": dm,
                    "Seed": SEED,
                })

            if kappa_idx % 10 == 0:
                print(f"    κ={kappa:.3f}: mean δ_min={np.mean(delta_mins):.4e}")

    df = pd.DataFrame(results)

    # Analyze sharpness: compute the steepness of δ_min vs κ transition per n
    print("\n--- Transition Sharpness Summary ---")
    for n in N_VALUES:
        sub = df[df["N"] == n]
        agg = sub.groupby("Kappa")["Delta_Min"].mean().reset_index()
        # Compute max gradient
        gradient = np.gradient(agg["Delta_Min"].values, agg["Kappa"].values)
        max_grad_idx = np.argmax(np.abs(gradient))
        print(f"  n={n}: max |gradient| = {np.abs(gradient[max_grad_idx]):.4f} "
              f"at κ={agg['Kappa'].values[max_grad_idx]:.3f}")

    save_results(df, "sharpness_vs_n.csv", SEED, "Sharpness vs n")


if __name__ == "__main__":
    run_experiment()
