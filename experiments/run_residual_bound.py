"""
experiments/run_residual_bound.py

Empirical vs. Theoretical Residual Bound Tightness

Uses measurement to inform the still-open τ(S) derivation.

Parameters:
  - 20 log-spaced perturbation sizes δ ∈ [1e-4, 1]
  - 300 trials per δ at fixed conditioning tiers
  - Seed: 42103
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    anti_circularity_check, fit_power_law_loglog, bootstrap_exponent_ci,
    random_six_tuple, RESULTS_DIR,
)
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual
from algorithms.conditioning.scores import compute_conditioning_score
from synthetic_data.generator import generate_generic_configuration


FINDING_ID = 3
SEED = get_finding_seed(FINDING_ID)  # 42103
N_TRIALS = 300
DELTA_GRID = np.logspace(-4, 0, 20)
N_POINTS = 12
LS_TIERS = {"well": (0, 100), "moderate": (100, 1000), "poor": (1000, 1e8)}


def run_experiment():
    print("=" * 60)
    print("EMPIRICAL RESIDUAL BOUND TIGHTNESS")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    for delta_idx, delta in enumerate(DELTA_GRID):
        for trial in range(N_TRIALS):
            trial_rng = make_rng(SEED + delta_idx * 10000 + trial)

            pts = generate_generic_configuration(N_POINTS, scale=1.0, rng=trial_rng)
            t = random_six_tuple(N_POINTS, trial_rng)

            # Compute L(S) for tier classification
            ls = compute_conditioning_score(pts, t)

            # Classify tier
            tier = "well"
            for tier_name, (lo, hi) in LS_TIERS.items():
                if lo <= ls < hi:
                    tier = tier_name
                    break

            # Perturb points by delta
            pts_perturbed = pts.copy()
            pts_perturbed[:3, :] += trial_rng.normal(0, delta, size=(3, N_POINTS))

            # Compute residual on perturbed points
            brackets = compute_all_brackets_for_tuple(pts_perturbed, t)
            rho = compute_syzygy_residual(brackets)

            results.append({
                "Delta": delta,
                "Trial": trial,
                "Rho": rho,
                "LS": ls,
                "LS_Tier": tier,
                "Seed": SEED,
            })

        if (delta_idx + 1) % 5 == 0:
            print(f"  δ step {delta_idx + 1}/20 done")

    df = pd.DataFrame(results)

    # Per-tier log-log fit
    print("\n--- Per-Tier Log-Log Fit: ρ vs δ ---")
    for tier in LS_TIERS:
        sub = df[df["LS_Tier"] == tier]
        if len(sub) < 100:
            print(f"  {tier}: insufficient data")
            continue
        agg = sub.groupby("Delta")["Rho"].mean().reset_index()
        result = fit_power_law_loglog(agg["Delta"].values, agg["Rho"].values)
        exp_est, ci_lo, ci_hi = bootstrap_exponent_ci(
            sub["Delta"].values, sub["Rho"].values, rng=make_rng(SEED + 999)
        )
        print(f"  {tier}: exponent={result['exponent']:.3f}, R²={result['r_squared']:.4f}, "
              f"CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

    save_results(df, "residual_bound.csv", SEED, "Residual bound tightness")


if __name__ == "__main__":
    run_experiment()
