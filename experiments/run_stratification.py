"""
experiments/run_stratification.py

stratification experiment — Stratification Confusion Matrix

Hypothesis: The stratification classifier correctly recovers ground-truth
stratum labels using bracket-magnitude thresholding.

Parameters:
  - 4 strata × 5,000 trials = 20,000 total
  - Threshold sweep: τ_strat ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
  - Seed: 3042

Independent generator check: for 200 configurations per stratum,
the size of the largest exactly coplanar subset (rank-3 test, tolerance 1e-8)
must be 0 (generic), 4 (boundary), 5 (compound) or 6 (critical). A check that
accepted any rank <= 4 for boundary/compound could not fail.

Second classifier: the rank-based classifier classify_stratum_by_rank, evaluated with the global
bracket-threshold classifier at noise levels sigma in {0, 1e-6, 1e-4} added to
the unit representatives, tolerance sqrt(6) * max(sigma, 1e-12). Output:
stratification_classifiers.csv.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_experiment_seed, make_rng, save_results, log_hardware_info,
    generate_stratum_config, anti_circularity_check,
    pick_first_six_tuple, RESULTS_DIR,
)
from algorithms.degeneracy.stratification import (
    classify_degeneracy_stratum,
    classify_stratum_by_rank,
    largest_coplanar_subset_size,
    DegeneracyStratum,
)
from algorithms.invariants.brackets import compute_bracket
from algorithms.utils import normalize_to_unit_sphere
from itertools import combinations


EXPERIMENT_ID = 3
SEED = get_experiment_seed(EXPERIMENT_ID)  # 3042
N_TRIALS = 5000
STRATA = ["generic", "boundary", "compound", "critical"]
THRESHOLD_SWEEP = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
NOISE_LEVELS = [0.0, 1e-6, 1e-4]


def run_experiment():
    print("=" * 60)
    print("EXP 3: STRATIFICATION CONFUSION MATRIX")
    print("=" * 60)

    hw_info = log_hardware_info()
    rng = make_rng(SEED)
    results = []
    rank_checks = []
    comparison = []

    strata_offsets = {"generic": 0, "boundary": 100000, "compound": 200000, "critical": 300000}

    for stratum in STRATA:
        print(f"  Generating {N_TRIALS} configs for stratum: {stratum}")
        for trial in range(N_TRIALS):
            trial_rng = make_rng(SEED + strata_offsets[stratum] + trial)
            pts = generate_stratum_config(stratum, n_points=6, rng=trial_rng)
            t = pick_first_six_tuple(6)

            # Compute all 15 bracket magnitudes
            pts_norm = normalize_to_unit_sphere(pts)
            bracket_mags = []
            for quad in combinations(range(6), 4):
                val = abs(compute_bracket(pts_norm, quad, pre_normalized=True))
                bracket_mags.append(val)
            min_bracket = min(bracket_mags)

            # Classify at each threshold
            for tau in THRESHOLD_SWEEP:
                predicted = classify_degeneracy_stratum(pts, t,
                                                        near_zero_threshold=tau,
                                                        boundary_threshold=tau * 100)
                results.append({
                    "True_Stratum": stratum,
                    "Predicted_Stratum": predicted.value,
                    "Threshold": tau,
                    "Min_Bracket_Mag": min_bracket,
                    "Trial": trial,
                    "Seed": SEED,
                })

            # Independent generator check: largest exactly coplanar subset.
            if trial < 200:
                m_exact = largest_coplanar_subset_size(pts, t, tolerance=1e-8)
                expected = {"generic": 0, "boundary": 4, "compound": 5, "critical": 6}[stratum]
                rank_checks.append({
                    "True_Stratum": stratum,
                    "Trial": trial,
                    "Largest_Coplanar_Subset": m_exact,
                    "Expected": expected,
                    "Check_Correct": m_exact == expected,
                })

            # Classifier comparison under noise (first 1,000 per stratum).
            if trial < 1000:
                unit = normalize_to_unit_sphere(pts)
                for noise in NOISE_LEVELS:
                    noisy = unit + noise * trial_rng.standard_normal(unit.shape)
                    tolerance = np.sqrt(6.0) * max(noise, 1e-12)
                    comparison.append({
                        "True_Stratum": stratum,
                        "Trial": trial,
                        "Noise_Sigma": noise,
                        "Threshold_Classifier": classify_degeneracy_stratum(
                            noisy, t, near_zero_threshold=1e-6, boundary_threshold=1e-4).value,
                        "Rank_Classifier": classify_stratum_by_rank(noisy, t, tolerance=tolerance).value,
                        "Rank_Tolerance": tolerance,
                        "Seed": SEED,
                    })

    df = pd.DataFrame(results)

    # ---- Confusion matrices per threshold ----
    print("\n--- Confusion Matrices per Threshold ---")
    for tau in THRESHOLD_SWEEP:
        sub = df[df["Threshold"] == tau]
        print(f"\n  Threshold τ = {tau}")
        confusion = pd.crosstab(sub["True_Stratum"], sub["Predicted_Stratum"],
                                margins=True)
        print(confusion.to_string())

        # Per-class precision/recall
        for stratum in STRATA:
            true_pos = len(sub[(sub["True_Stratum"] == stratum) &
                              (sub["Predicted_Stratum"] == stratum)])
            total_true = len(sub[sub["True_Stratum"] == stratum])
            total_pred = len(sub[sub["Predicted_Stratum"] == stratum])
            recall = true_pos / max(total_true, 1)
            precision = true_pos / max(total_pred, 1)
            print(f"    {stratum}: precision={precision:.3f}, recall={recall:.3f}")

        # Overall accuracy
        correct = len(sub[sub["True_Stratum"] == sub["Predicted_Stratum"]])
        accuracy = correct / len(sub)
        print(f"    Overall accuracy: {accuracy:.3f}")

    # ---- Independent generator check ----
    print("\n--- Independent Check: largest exactly coplanar subset ---")
    rank_df = pd.DataFrame(rank_checks)
    for stratum in STRATA:
        sub = rank_df[rank_df["True_Stratum"] == stratum]
        print(f"  {stratum}: {int(sub['Check_Correct'].sum())}/{len(sub)} correct")

    # ---- Classifier comparison ----
    print("\n--- Accuracy: global threshold vs rank-based classifier ---")
    comparison_df = pd.DataFrame(comparison)
    for noise, group in comparison_df.groupby("Noise_Sigma"):
        acc_threshold = (group["Threshold_Classifier"] == group["True_Stratum"]).mean()
        acc_rank = (group["Rank_Classifier"] == group["True_Stratum"]).mean()
        print(f"  sigma={noise:.0e}: threshold {acc_threshold:.3f}, rank {acc_rank:.3f}")
        for stratum in STRATA:
            g = group[group["True_Stratum"] == stratum]
            print(f"     {stratum:9s} recall: threshold {(g['Threshold_Classifier'] == stratum).mean():.3f}, "
                  f"rank {(g['Rank_Classifier'] == stratum).mean():.3f}")

    # ---- Anti-circularity ----
    print("\n--- Anti-Circularity Check ---")
    for stratum in STRATA:
        sub = df[(df["True_Stratum"] == stratum) & (df["Threshold"] == 1e-4)]
        anti_circularity_check(sub["Min_Bracket_Mag"].values,
                               label=f"{stratum}/min_bracket")

    save_results(df, "stratification_confusion.csv", SEED,
                 "stratification experiment: Stratification Confusion Matrix")
    save_results(rank_df, "stratification_rank_checks.csv", SEED,
                 "stratification experiment: Independent generator check (largest coplanar subset)")
    save_results(comparison_df, "stratification_classifiers.csv", SEED,
                 "stratification experiment: Global-threshold vs rank-based stratum classifier under noise")


if __name__ == "__main__":
    run_experiment()
