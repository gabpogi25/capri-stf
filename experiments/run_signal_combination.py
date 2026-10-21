"""
experiments/run_signal_combination.py

Functional Form of R(p) (Signal Combination)

Compares arithmetic mean, geometric mean, min, and learned fusion
as R(p) aggregation strategies.

Parameters:
  - Same dataset as fusion experiment (if available), otherwise synthetic
  - 4 candidate forms × 5-fold CV
  - Seed: 42106
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    MEASUREMENT_NOISE_SIGMA,
    get_finding_seed, make_rng, save_results, log_hardware_info,
    bootstrap_ci, random_six_tuple, RESULTS_DIR,
)
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual
from algorithms.conditioning.scores import compute_conditioning_score
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_mismatched_configuration,
    measure_with_gaussian_noise,
    estimate_brackets_independently,
)


FINDING_ID = 6
SEED = get_finding_seed(FINDING_ID)  # 42106
N_SAMPLES = 10000
N_POINTS = 12
N_FOLDS = 5


def run_experiment():
    print("=" * 60)
    print("FUNCTIONAL FORM OF R(p)")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)

    # Generate labeled dataset: L(S), ρ(S), desc_confidence, ground_truth_correct
    print("  Generating labeled dataset...")
    data = []
    regimes = ["Generic", "Generic", "Mismatched", "Mismatched", "Generic"]

    for i in range(N_SAMPLES):
        trial_rng = make_rng(SEED + i)
        is_mismatch = trial_rng.random() < 0.4

        # Two noisy measurements (sigma = 1e-3); for mismatches measurement B
        # has 20% of points replaced. Each bracket comes from one measurement,
        # so rho can respond to the mismatch (coherent recomputation cannot).
        if is_mismatch:
            pts, pts_b, _ = generate_mismatched_configuration(
                N_POINTS, mismatch_fraction=0.2, scale=1.0, rng=trial_rng
            )
            correct = 0
        else:
            pts = generate_generic_configuration(N_POINTS, scale=1.0, rng=trial_rng)
            pts_b = pts
            correct = 1

        t = random_six_tuple(N_POINTS, trial_rng)
        measurements = [measure_with_gaussian_noise(pts, MEASUREMENT_NOISE_SIGMA, trial_rng),
                        measure_with_gaussian_noise(pts_b, MEASUREMENT_NOISE_SIGMA, trial_rng)]
        brackets, _ = estimate_brackets_independently(measurements, t, trial_rng)
        rho = compute_syzygy_residual(brackets)
        ls = compute_conditioning_score(pts, t)

        data.append({
            "Sample": i,
            "LS": ls,
            "Rho": rho,
            "Correct": correct,
        })

    df = pd.DataFrame(data)

    # Normalize signals to [0, 1]
    ls_vals = df["LS"].values.copy()
    ls_vals[~np.isfinite(ls_vals)] = np.nanmax(ls_vals[np.isfinite(ls_vals)])
    ls_norm = 1.0 - (ls_vals - ls_vals.min()) / (ls_vals.max() - ls_vals.min() + 1e-30)

    rho_vals = df["Rho"].values.copy()
    rho_norm = 1.0 - (rho_vals - rho_vals.min()) / (rho_vals.max() - rho_vals.min() + 1e-30)

    y = df["Correct"].values

    # ---- Evaluate combination forms (only real geometric signals) ----
    forms = {
        "Arithmetic Mean": (ls_norm + rho_norm) / 2.0,
        "Geometric Mean": np.sqrt(np.maximum(ls_norm, 1e-10) *
                                   np.maximum(rho_norm, 1e-10)),
        "Min": np.minimum(ls_norm, rho_norm),
        "LS Only": ls_norm,
        "Rho Only": rho_norm,
    }

    print("\n--- AUC Comparison (5-fold CV) ---")
    for form_name, scores in forms.items():
        fold_aucs = []
        indices = np.arange(len(y))
        rng.shuffle(indices)
        fold_size = len(indices) // N_FOLDS

        for fold in range(N_FOLDS):
            test_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            y_test = y[test_idx]
            s_test = scores[test_idx]

            # Compute AUC via trapezoidal
            thresholds = np.linspace(0, 1, 100)
            tprs, fprs = [], []
            for thresh in thresholds:
                pred = (s_test >= thresh).astype(int)
                tp = np.sum((pred == 1) & (y_test == 1))
                fp = np.sum((pred == 1) & (y_test == 0))
                fn = np.sum((pred == 0) & (y_test == 1))
                tn = np.sum((pred == 0) & (y_test == 0))
                tprs.append(tp / max(tp + fn, 1))
                fprs.append(fp / max(fp + tn, 1))
            auc = abs(float(np.trapezoid(tprs, fprs)))
            fold_aucs.append(auc)

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"  {form_name:20s}: AUC = {mean_auc:.4f} ± {std_auc:.4f}")

    # ---- Brier score ----
    print("\n--- Brier Score Comparison ---")
    for form_name, scores in forms.items():
        brier = np.mean((scores - y) ** 2)
        print(f"  {form_name:20s}: Brier = {brier:.4f}")

    save_results(df, "signal_combination.csv", SEED, "Signal combination")


if __name__ == "__main__":
    run_experiment()
