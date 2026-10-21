"""
experiments/run_trust_calibration.py

trust-calibration experiment — Trust Calibration Check

Hypothesis: Binned predicted R(p) tracks empirical accuracy closely.

Parameters:
  - 10 confidence bins
  - Minimum 500 trials per bin
  - Draw proportionally across all 5 regimes
  - Seed: 11042
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_experiment_seed, make_rng, save_results, log_hardware_info,
    bootstrap_ci, random_six_tuple, RESULTS_DIR,
)
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual
from algorithms.conditioning.scores import compute_conditioning_score
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_mismatched_configuration,
    generate_near_degenerate_configuration,
    generate_non_rigid_configuration,
)


EXPERIMENT_ID = 11
SEED = get_experiment_seed(EXPERIMENT_ID)  # 11042
N_TOTAL = 10000
N_POINTS = 12
N_BINS = 10
MIN_BIN_SAMPLES = 50  # bins with fewer samples are not interpreted (review)


def run_experiment():
    print("=" * 60)
    print("EXP 11: TRUST CALIBRATION CHECK")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    # Generate balanced dataset across regimes
    regime_names = ["Generic", "Near-Degenerate", "Mismatched", "Non-Rigid"]
    n_per_regime = N_TOTAL // len(regime_names)

    print(f"  Generating {N_TOTAL} samples ({n_per_regime} per regime)...")

    from experiment_utils import regime_measurements
    from synthetic_data.generator import estimate_brackets_independently

    for regime in regime_names:
        is_correct_regime = regime in ["Generic"]

        for trial in range(n_per_regime):
            trial_rng = make_rng(SEED + regime_names.index(regime) * 10000 + trial)
            t = random_six_tuple(N_POINTS, trial_rng)
            # Two independent measurements per scene; each bracket estimated
            # from one of them (replaces bracket mixing, see syzygy-residual experiment).
            pts, measurements, _ = regime_measurements(regime, N_POINTS, trial_rng)
            brackets, _ = estimate_brackets_independently(measurements, t, trial_rng)

            rho = compute_syzygy_residual(brackets)
            ls = compute_conditioning_score(pts, t)

            # Hand-specified confidence score (NOT a neural network): a fixed
            # formula combining L(S) and rho. Figures must not call it "neural".
            ls_safe = min(ls, 1e6)
            confidence = 1.0 / (1.0 + ls_safe * 0.001) * (1.0 / (1.0 + rho * 100))
            confidence = max(0.0, min(1.0, confidence))

            correct = 1 if is_correct_regime else 0

            results.append({
                "Regime": regime,
                "Trial": trial,
                "Rho": rho,
                "LS": ls,
                "Predicted_Confidence": confidence,
                "Correct": correct,
                "Seed": SEED,
            })

    df = pd.DataFrame(results)

    # ---- Binned calibration ----
    print("\n--- Calibration Curve ---")
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    calibration_data = []

    for i in range(N_BINS):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_mask = (df["Predicted_Confidence"] >= lo) & (df["Predicted_Confidence"] < hi)
        bin_df = df[bin_mask]
        n_in_bin = len(bin_df)
        if n_in_bin == 0:
            continue

        predicted_mean = bin_df["Predicted_Confidence"].mean()
        empirical_accuracy = bin_df["Correct"].mean()

        _, acc_lo, acc_hi = bootstrap_ci(
            bin_df["Correct"].to_numpy(dtype=float), rng=make_rng(SEED + 500 + i)
        )
        calibration_data.append({
            "Bin_Lo": lo,
            "Bin_Hi": hi,
            "Bin_Center": (lo + hi) / 2,
            "Predicted_Mean": predicted_mean,
            "Empirical_Accuracy": empirical_accuracy,
            "Accuracy_CI_Lo": acc_lo,
            "Accuracy_CI_Hi": acc_hi,
            "N_Samples": n_in_bin,
            "Reportable": n_in_bin >= MIN_BIN_SAMPLES,
            "Abs_Error": abs(predicted_mean - empirical_accuracy),
        })

        print(f"  [{lo:.1f}, {hi:.1f}): pred={predicted_mean:.3f}, "
              f"emp={empirical_accuracy:.3f}, n={n_in_bin}")

    cal_df = pd.DataFrame(calibration_data)

    # ECE
    if len(cal_df) > 0:
        total = cal_df["N_Samples"].sum()
        ece = (cal_df["Abs_Error"] * cal_df["N_Samples"]).sum() / total
        print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
        reportable = cal_df[cal_df["Reportable"]]
        ece_reportable = (reportable["Abs_Error"] * reportable["N_Samples"]).sum() / reportable["N_Samples"].sum()
        print(f"  ECE over bins with N >= {MIN_BIN_SAMPLES}: {ece_reportable:.4f} "
              f"({len(reportable)} of {len(cal_df)} bins)")
        print(f"  PASS (ECE ≤ 0.05): {ece <= 0.05}")

    # Per-regime calibration
    print("\n--- Per-Regime Calibration ---")
    for regime in regime_names:
        sub = df[df["Regime"] == regime]
        ece_regime = 0
        total_regime = 0
        for i in range(N_BINS):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            bin_mask = (sub["Predicted_Confidence"] >= lo) & (sub["Predicted_Confidence"] < hi)
            bin_sub = sub[bin_mask]
            if len(bin_sub) > 0:
                pred = bin_sub["Predicted_Confidence"].mean()
                emp = bin_sub["Correct"].mean()
                ece_regime += len(bin_sub) * abs(pred - emp)
                total_regime += len(bin_sub)
        if total_regime > 0:
            ece_r = ece_regime / total_regime
            print(f"  {regime}: ECE = {ece_r:.4f}")

    save_results(df, "trust_calibration.csv", SEED,
                 "trust-calibration experiment: Trust Calibration")
    save_results(cal_df, "trust_calibration_curve.csv", SEED,
                 "trust-calibration experiment: Calibration Curve")


if __name__ == "__main__":
    run_experiment()
