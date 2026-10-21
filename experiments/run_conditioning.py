"""
experiments/run_conditioning.py

conditioning experiment — Conditioning Score ROC + Greedy vs. Random Selection

Hypothesis: L(S) separates well-conditioned from poorly-conditioned tuples
with strong ROC performance, and greedy tuple selection outperforms random.

Parameters:
  ROC half:
    - 10,000 six-point tuples: 5,000 near-degenerate, 5,000 generic
    - 50 log-spaced thresholds
  Greedy half:
    - Universe sizes N ∈ {1000, 5000, 10000}
    - Budget k ∈ {10, 20, ..., 100}
    - For N ≤ 15: exhaustive OPT search
    - 100 repetitions for random (stochastic)
  - Seed: 4042
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_experiment_seed, make_rng, save_results, log_hardware_info,
    generate_stratum_config, generate_near_degenerate_config,
    anti_circularity_check, bootstrap_ci, pick_first_six_tuple,
    random_six_tuple, RESULTS_DIR,
)
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.selection.greedy import select_tuples_greedy
from synthetic_data.generator import generate_generic_configuration


EXPERIMENT_ID = 4
SEED = get_experiment_seed(EXPERIMENT_ID)  # 4042


def compute_coverage(selected_tuples: list, n_points: int) -> float:
    """Coverage objective: fraction of points covered by at least one tuple."""
    covered = set()
    for t in selected_tuples:
        covered.update(t)
    return len(covered) / n_points


def run_roc_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """ROC half: evaluate L(S) as a classifier for degenerate vs generic."""
    print("\n--- ROC Experiment ---")
    N_POS = 5000  # Near-degenerate (poorly conditioned)
    N_NEG = 5000  # Generic (well conditioned)

    scores = []
    labels = []

    # Positive class: near-degenerate
    print("  Generating 5,000 near-degenerate tuples...")
    for i in range(N_POS):
        kappa = rng.uniform(0.99, 0.999)
        pts = generate_near_degenerate_config(6, kappa=kappa, rng=rng)
        t = pick_first_six_tuple(6)
        ls = compute_conditioning_score(pts, t)
        scores.append(ls)
        labels.append(1)  # Positive = poorly conditioned

    # Negative class: generic
    print("  Generating 5,000 generic tuples...")
    for i in range(N_NEG):
        pts = generate_stratum_config("generic", n_points=6, rng=rng)
        t = pick_first_six_tuple(6)
        ls = compute_conditioning_score(pts, t)
        scores.append(ls)
        labels.append(0)  # Negative = well conditioned

    scores = np.array(scores)
    labels = np.array(labels)

    # Threshold sweep
    finite_scores = scores[np.isfinite(scores)]
    if len(finite_scores) == 0:
        print("  WARNING: All scores are infinite!")
        return pd.DataFrame()

    thresholds = np.logspace(
        np.log10(max(finite_scores.min(), 1e-6)),
        np.log10(finite_scores.max()),
        50
    )

    roc_data = []
    for thresh in thresholds:
        predicted_positive = scores >= thresh
        tp = np.sum(predicted_positive & (labels == 1))
        fp = np.sum(predicted_positive & (labels == 0))
        fn = np.sum(~predicted_positive & (labels == 1))
        tn = np.sum(~predicted_positive & (labels == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        roc_data.append({
            "Threshold": thresh,
            "TPR": tpr,
            "FPR": fpr,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Seed": SEED,
        })

    df_roc = pd.DataFrame(roc_data)

    # AUC (trapezoidal)
    sorted_roc = df_roc.sort_values("FPR")
    auc = float(np.trapezoid(sorted_roc["TPR"].values, sorted_roc["FPR"].values))
    print(f"  ROC AUC: {auc:.4f}")
    print(f"  PASS: {auc >= 0.90}")

    return df_roc


def run_greedy_experiment(rng: np.random.Generator) -> pd.DataFrame:
    """Greedy vs Random selection at matched budgets."""
    print("\n--- Greedy vs Random Experiment ---")

    budgets = list(range(10, 101, 10))
    n_cloud = 30  # Point cloud size
    n_random_reps = 100
    n_cloud_reps = 10  # Multiple independent clouds for statistical power
    greedy_results = []

    for cloud_rep in range(n_cloud_reps):
        cloud_rng = make_rng(SEED + cloud_rep * 7777)
        pts = generate_generic_configuration(n_cloud, scale=1.0, rng=cloud_rng)

        for k in budgets:
            if cloud_rep == 0:
                print(f"  Budget k={k}...")

            # Greedy selection (deterministic given seed)
            greedy_tuples = select_tuples_greedy(
                pts,
                conditioning_threshold=1e6,
                max_tuples=k,
                rng=make_rng(SEED + cloud_rep * 10000 + k),
            )
            greedy_coverage = compute_coverage(greedy_tuples, n_cloud)

            # Random selection: 100 repetitions
            random_coverages = []
            
            # Precompute candidate index pool once to avoid combinatorial explosion inside the loop
            all_possible = list(combinations(range(n_cloud), 6))
            
            for rep in range(n_random_reps):
                random_rng = make_rng(SEED + cloud_rep * 100000 + k * 1000 + rep)
                # Select k random tuples
                if len(all_possible) > 10000:
                    sample_idx = random_rng.choice(len(all_possible), size=min(10000, len(all_possible)), replace=False)
                    candidates = [all_possible[i] for i in sample_idx]
                else:
                    candidates = all_possible
                selected_idx = random_rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
                random_tuples = [candidates[i] for i in selected_idx]
                random_coverages.append(compute_coverage(random_tuples, n_cloud))

            random_mean = float(np.mean(random_coverages))
            random_std = float(np.std(random_coverages))
            theoretical_floor = 1.0 - (1.0 / np.e)  # approx 0.632

            greedy_results.append({
                "Cloud_Rep": cloud_rep,
                "Budget_k": k,
                "Greedy_Coverage": greedy_coverage,
                "Random_Mean_Coverage": random_mean,
                "Random_Std_Coverage": random_std,
                "Theoretical_Floor": theoretical_floor,
                "Greedy_Random_Ratio": greedy_coverage / max(random_mean, 1e-10),
                "N_Cloud": n_cloud,
                "Seed": SEED,
            })

    df_greedy = pd.DataFrame(greedy_results)
    print("\n  Greedy vs Random Summary (averaged over clouds):")
    agg = df_greedy.groupby("Budget_k").agg({
        "Greedy_Coverage": "mean",
        "Random_Mean_Coverage": "mean",
        "Theoretical_Floor": "first",
        "Greedy_Random_Ratio": "mean",
    }).reset_index()
    print(agg.to_string(index=False))

    return df_greedy


def run_small_n_opt(rng: np.random.Generator) -> pd.DataFrame:
    """For N ≤ 15, compute exact OPT by exhaustive search."""
    print("\n--- Small-N Exhaustive OPT Search ---")

    n_cloud = 8  # C(8,6) = 28, making C(28,k) feasible for small k
    budgets = [2, 3, 4, 5]  # C(28,5) = 98,280, just under 100k limit
    opt_results = []

    pts = generate_generic_configuration(n_cloud, scale=1.0, rng=rng)
    all_tuples = list(combinations(range(n_cloud), 6))
    print(f"  N={n_cloud}, |all_tuples|={len(all_tuples)}")

    for k in budgets:
        print(f"  Computing OPT for k={k}...")
        import math
        num_combinations = math.comb(len(all_tuples), k)
        
        # Exhaustive search: try all C(|all_tuples|, k) subsets
        # Only feasible if total combinations is reasonable (<100,000).
        if num_combinations <= 100000:
            best_coverage = 0.0
            for subset in combinations(range(len(all_tuples)), k):
                selected = [all_tuples[i] for i in subset]
                cov = compute_coverage(selected, n_cloud)
                if cov > best_coverage:
                    best_coverage = cov
            opt_coverage = best_coverage
        else:
            # Too many combinations (e.g. 924 choose 5 is 6 trillion!)
            # Use upper bound instead of hanging forever
            opt_coverage = 1.0  # Trivial upper bound

        # Greedy
        greedy_tuples = select_tuples_greedy(
            pts, conditioning_threshold=1e6, max_tuples=k,
            rng=make_rng(SEED + k + 5000),
        )
        greedy_coverage = compute_coverage(greedy_tuples, n_cloud)

        opt_results.append({
            "N": n_cloud,
            "Budget_k": k,
            "OPT_Coverage": opt_coverage,
            "Greedy_Coverage": greedy_coverage,
            "Greedy_OPT_Ratio": greedy_coverage / max(opt_coverage, 1e-10),
            "Seed": SEED,
        })

    df_opt = pd.DataFrame(opt_results)
    print(df_opt.to_string(index=False))
    return df_opt


def run_experiment():
    print("=" * 60)
    print("EXP 4: CONDITIONING SCORE ROC + GREEDY SELECTION")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)

    df_roc = run_roc_experiment(rng)
    df_greedy = run_greedy_experiment(rng)
    df_opt = run_small_n_opt(rng)

    save_results(df_roc, "conditioning_roc.csv", SEED, "conditioning experiment: ROC")
    save_results(df_greedy, "conditioning_greedy_vs_random.csv", SEED, "conditioning experiment: Greedy vs Random")
    save_results(df_opt, "conditioning_small_n_opt.csv", SEED, "conditioning experiment: Small-N OPT")


if __name__ == "__main__":
    run_experiment()
