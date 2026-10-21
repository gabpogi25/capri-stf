"""
experiments/run_drift_admission.py

Drift admissibility sweep on the
bounded-perturbation model of Theorem 6.1 (docs/MATHEMATICS.md).

Model used here (the hypothesis of Theorem 6.1): two measurements of the same
six points, each obtained by moving every unit representative by a vector of
norm at most r. Each of the 15 brackets is estimated from one of the two
measurements, chosen at random.
  * Bounded condition: r = eta. Theorem 6.1 PROVES rho <= tau_exact(eta), so the
    admission rate must be exactly 100%; any violation would be a bug.
  * Excessive condition: r = c * eta with c ~ U(1.5, 5). The test is whether
    rho rises above tau_exact(eta), i.e. whether the worst-case bound ever
    rejects. The reported Ratio = rho / tau_exact(eta) shows how far below the
    bound typical residuals sit (the bound is worst case, so low rejection is
    expected and is reported as such).

Parameters: 50 log-spaced eta in [0.001, 0.2], 300 trials per eta, 12 points.
Seed 42006.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import get_experiment_seed, make_rng, save_results, random_six_tuple
from synthetic_data.generator import (
    generate_generic_configuration,
    perturb_within_ball,
    estimate_brackets_independently,
)
from algorithms.syzygy.residual import (
    compute_syzygy_residual,
    compute_exact_admissibility_bound,
    compute_first_order_bound,
)
from algorithms.conditioning.scores import compute_conditioning_score

EXPERIMENT_ID = 6
SEED = get_experiment_seed(EXPERIMENT_ID)  # 42006
N_TRIALS_PER_ETA = 300
N_POINTS = 12
ETAS = np.logspace(np.log10(0.001), np.log10(0.2), 50)


def _residual_for_radius(points: np.ndarray, radius: float, t: tuple, rng: np.random.Generator) -> float:
    measurements = [perturb_within_ball(points, radius, rng), perturb_within_ball(points, radius, rng)]
    brackets, _ = estimate_brackets_independently(measurements, t, rng)
    return compute_syzygy_residual(brackets)


def run_experiment() -> pd.DataFrame:
    print("=" * 60)
    print("EXP 6: DRIFT ADMISSIBILITY SWEEP (Theorem 6.1 model)")
    print("=" * 60)
    rows = []
    for eta_index, eta in enumerate(ETAS):
        tau_exact = compute_exact_admissibility_bound(eta)
        for trial in range(N_TRIALS_PER_ETA):
            trial_rng = make_rng(SEED + eta_index * 10000 + trial)
            points = generate_generic_configuration(N_POINTS, scale=1.0, rng=trial_rng)
            t = random_six_tuple(N_POINTS, trial_rng)
            excess_factor = float(trial_rng.uniform(1.5, 5.0))
            rho_bounded = _residual_for_radius(points, eta, t, trial_rng)
            rho_excessive = _residual_for_radius(points, excess_factor * eta, t, trial_rng)
            rows.append({
                "Eta": eta,
                "Trial": trial,
                "Excess_Factor": excess_factor,
                "LS": compute_conditioning_score(points, t),
                "Tau_Exact": tau_exact,
                "Tau_First_Order": compute_first_order_bound(eta),
                "Rho_Bounded": rho_bounded,
                "Rho_Excessive": rho_excessive,
                "Ratio_Bounded": rho_bounded / tau_exact,
                "Ratio_Excessive": rho_excessive / tau_exact,
                "Admitted_Bounded": bool(rho_bounded <= tau_exact),
                "Rejected_Excessive": bool(rho_excessive > tau_exact),
                "Seed": SEED,
            })
    df = pd.DataFrame(rows)

    print(f"  Bounded admission rate:   {df['Admitted_Bounded'].mean():.4f} (theorem requires 1.0)")
    print(f"  Excessive rejection rate: {df['Rejected_Excessive'].mean():.4f}")
    print(f"  Median rho/tau (bounded):   {df['Ratio_Bounded'].median():.3e}")
    print(f"  Median rho/tau (excessive): {df['Ratio_Excessive'].median():.3e}")
    print(f"  Max rho/tau (bounded):      {df['Ratio_Bounded'].max():.3e}")
    assert df["Admitted_Bounded"].all(), "Theorem 6.1 violated: check the implementation."

    # Calibrated alternative: an empirical-null threshold.
    # Per eta, the 95th percentile of rho over the first half of the bounded
    # trials (calibration split) is applied to the second half (test split).
    calibrated_rows = []
    for eta, group in df.groupby("Eta"):
        calibration = group[group["Trial"] < N_TRIALS_PER_ETA // 2]
        test = group[group["Trial"] >= N_TRIALS_PER_ETA // 2]
        threshold = float(np.quantile(calibration["Rho_Bounded"], 0.95))
        calibrated_rows.append({
            "Eta": eta,
            "Threshold_Q95": threshold,
            "Tau_Exact": float(group["Tau_Exact"].iloc[0]),
            "False_Rejection_Bounded": float((test["Rho_Bounded"] > threshold).mean()),
            "Rejection_Excessive": float((test["Rho_Excessive"] > threshold).mean()),
            "N_Test": int(len(test)),
            "Seed": SEED,
        })
    calibrated = pd.DataFrame(calibrated_rows)
    print(f"  Calibrated q95 threshold: false rejection {calibrated['False_Rejection_Bounded'].mean():.3f}, "
          f"excessive rejection {calibrated['Rejection_Excessive'].mean():.3f} (mean over eta)")

    save_results(df, "drift_admission.csv", SEED, "drift-admission experiment: Drift admissibility under Theorem 6.1 model")
    save_results(calibrated, "drift_admission_calibrated.csv", SEED,
                 "drift-admission experiment: Empirical-null (q95) threshold, calibration/test split per eta")
    return df


if __name__ == "__main__":
    run_experiment()
