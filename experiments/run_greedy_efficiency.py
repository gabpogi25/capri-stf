"""
experiments/run_greedy_efficiency.py

Greedy Selection Efficiency Ratio

Directly extends conditioning experiment's greedy/OPT comparison.

What is and is not an exact optimum
-----------------------------------
* n = 10: OPT is computed exactly (OPT_Method = "exhaustive") by breadth-first
  search over the 2^10 coverage bitmasks: from the empty set, each step adds
  one feasible tuple (L(S) below the same threshold greedy uses); OPT(k) is
  the largest coverage reachable in at most k steps. This is exact and takes
  milliseconds. (Enumerating k-subsets of the 210 tuples is not feasible:
  C(210, 3) > 1.5 million.)
* n = 50: OPT is never computed; only the upper bound 1 is used.
Greedy_OPT_Ratio is filled only for exhaustive rows. Greedy_UpperBound_Ratio
(= greedy coverage) is filled for every row. There is no n = 100 and no
exact optimum beyond n = 10.

Why greedy can fall below 1 - 1/e here: the objective is
f(G) = cov(G) - 0.001 * sum L(t). Tuples whose penalty exceeds their coverage
gain have negative marginal gain, and greedy stops adding tuples (N_Selected
records how many it chose). These are consequences of the non-monotone
penalty term, not evidence that coverage itself is non-submodular.

Coverage selector (docs/MATHEMATICS.md, Section 5; negative finding N4)
----------------------------------------------------------------------
Columns MaxCov_*: the pipeline's selector, greedy maximum coverage over the
tuples with kappa_J < KAPPA_THRESHOLD (algorithms/selection/coverage.py).
KAPPA_THRESHOLD = 2 sqrt(2 * CONDITIONING_THRESHOLD), so by Proposition 5.2
every tuple feasible for the L(S) rule is feasible here. Its exact optimum
(MaxCov_OPT_Coverage) is computed over the same kappa-feasible set, so
MaxCov_OPT_Ratio >= 1 - 1/e is guaranteed (Nemhauser-Wolsey-Fisher). The
selector uses its own seeded generator, so it does not change the
penalised-greedy columns.

Parameters:
  - Shares conditioning experiment methodology
  - Seed: 42105
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    bootstrap_ci, RESULTS_DIR,
)
from algorithms.selection.greedy import select_tuples_greedy
from algorithms.conditioning.scores import (
    batch_invariant_noise_gain, compute_conditioning_score, compute_invariant_noise_gain,
)
from algorithms.selection.coverage import select_tuples_max_coverage
from synthetic_data.generator import generate_generic_configuration


FINDING_ID = 5
SEED = get_finding_seed(FINDING_ID)  # 42105


def compute_coverage_objective(tuples: list, n_points: int) -> float:
    covered = set()
    for t in tuples:
        covered.update(t)
    return len(covered) / n_points


CONDITIONING_THRESHOLD = 1e6
KAPPA_THRESHOLD = 2.0 * np.sqrt(2.0 * CONDITIONING_THRESHOLD)


def max_coverage_selection(pts: np.ndarray, k: int, seed: int) -> list:
    selected, _ = select_tuples_max_coverage(pts, compute_invariant_noise_gain, KAPPA_THRESHOLD, max_tuples=k,
                                             rng=make_rng(seed), batch_score_fn=batch_invariant_noise_gain)
    return selected


def exact_max_coverage(feasible_tuples: list, n_points: int, budgets: list) -> dict:
    """Exact max coverage for each budget k by BFS over coverage bitmasks."""
    masks = sorted({sum(1 << i for i in t) for t in feasible_tuples})
    frontier, seen, best, results = {0}, {0}, 0, {}
    for step in range(1, max(budgets) + 1):
        new = set()
        for state in frontier:
            for m in masks:
                nxt = state | m
                if nxt not in seen:
                    seen.add(nxt)
                    new.add(nxt)
        if new:
            best = max(best, max(bin(x).count("1") for x in new))
        frontier = new or frontier
        if step in budgets:
            results[step] = best / n_points
    return results


def run_experiment():
    print("=" * 60)
    print("GREEDY SELECTION EFFICIENCY RATIO")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    # Small-N: exhaustive OPT for n=10
    n_small = 10
    budgets = [3, 5, 8, 10]
    n_repetitions = 50

    print(f"\n--- Small-N Exhaustive OPT (n={n_small}) ---")
    for rep in range(n_repetitions):
        trial_rng = make_rng(SEED + rep)
        pts = generate_generic_configuration(n_small, scale=1.0, rng=trial_rng)
        all_tuples = list(combinations(range(n_small), 6))
        feasible = [t for t in all_tuples
                    if compute_conditioning_score(pts, t) < CONDITIONING_THRESHOLD]
        opt_by_budget = exact_max_coverage(feasible, n_small, budgets)
        kappa = batch_invariant_noise_gain(pts, np.array(all_tuples, dtype=np.int64))
        feasible_kappa = [t for t, value in zip(all_tuples, kappa) if value < KAPPA_THRESHOLD]
        opt_kappa_by_budget = exact_max_coverage(feasible_kappa, n_small, budgets)

        for k in budgets:
            # Greedy
            greedy_sel = select_tuples_greedy(
                pts, conditioning_threshold=CONDITIONING_THRESHOLD, max_tuples=k, rng=trial_rng
            )
            greedy_cov = compute_coverage_objective(greedy_sel, n_small)

            best_opt = opt_by_budget[k]
            opt_method = "exhaustive"

            theoretical_floor = 1.0 - 1.0 / np.e
            ratio = greedy_cov / best_opt if opt_method == "exhaustive" else float("nan")
            maxcov_sel = max_coverage_selection(pts, k, SEED + 70000 + rep)
            maxcov_cov = compute_coverage_objective(maxcov_sel, n_small)

            results.append({
                "N": n_small,
                "Budget_k": k,
                "Rep": rep,
                "N_Selected": len(greedy_sel),
                "Greedy_Coverage": greedy_cov,
                "OPT_Coverage": best_opt,
                "OPT_Method": opt_method,
                "Greedy_OPT_Ratio": ratio,
                "Greedy_UpperBound_Ratio": greedy_cov,
                "Theoretical_Floor": theoretical_floor,
                "Above_Floor": bool(ratio >= theoretical_floor) if opt_method == "exhaustive" else None,
                "MaxCov_N_Selected": len(maxcov_sel),
                "MaxCov_Coverage": maxcov_cov,
                "MaxCov_OPT_Coverage": opt_kappa_by_budget[k],
                "MaxCov_OPT_Ratio": maxcov_cov / opt_kappa_by_budget[k],
                "Seed": SEED,
            })

        if (rep + 1) % 10 == 0:
            print(f"  Rep {rep + 1}/{n_repetitions}")

    # Large-N: greedy vs random (n=50)
    n_large = 50
    budgets_large = [10, 20, 50, 100]
    print(f"\n--- Large-N Greedy vs Random (n={n_large}) ---")

    for rep in range(20):
        trial_rng = make_rng(SEED + 50000 + rep)
        pts = generate_generic_configuration(n_large, scale=1.0, rng=trial_rng)

        for k in budgets_large:
            greedy_sel = select_tuples_greedy(
                pts, conditioning_threshold=CONDITIONING_THRESHOLD, max_tuples=k, rng=trial_rng
            )
            greedy_cov = compute_coverage_objective(greedy_sel, n_large)
            maxcov_sel = max_coverage_selection(pts, k, SEED + 80000 + rep)

            # Random baseline
            random_covs = []
            for r in range(50):
                r_rng = make_rng(SEED + 60000 + rep * 100 + r)
                indices = np.array([np.sort(r_rng.choice(n_large, size=6, replace=False)) for _ in range(k)])
                random_tuples = [tuple(row) for row in indices]
                random_covs.append(compute_coverage_objective(random_tuples, n_large))
            random_mean = np.mean(random_covs)

            results.append({
                "N": n_large,
                "Budget_k": k,
                "Rep": rep,
                "N_Selected": len(greedy_sel),
                "Greedy_Coverage": greedy_cov,
                "OPT_Coverage": float("nan"),
                "OPT_Method": "upper_bound",
                "Greedy_OPT_Ratio": float("nan"),
                "Greedy_UpperBound_Ratio": greedy_cov,
                "Theoretical_Floor": 1.0 - 1.0 / np.e,
                "Above_Floor": None,
                "Random_Mean": random_mean,
                "MaxCov_N_Selected": len(maxcov_sel),
                "MaxCov_Coverage": compute_coverage_objective(maxcov_sel, n_large),
                "MaxCov_OPT_Coverage": float("nan"),
                "MaxCov_OPT_Ratio": float("nan"),
                "Seed": SEED,
            })

    df = pd.DataFrame(results)

    print("\n--- Summary ---")
    for n in [n_small, n_large]:
        sub = df[df["N"] == n]
        for k in sub["Budget_k"].unique():
            sub_k = sub[sub["Budget_k"] == k]
            exact = sub_k[sub_k["OPT_Method"] == "exhaustive"]
            if len(exact):
                print(f"  n={n}, k={k}: greedy/OPT mean={exact['Greedy_OPT_Ratio'].mean():.3f}, "
                      f"below (1-1/e): {int((exact['Greedy_OPT_Ratio'] < 1 - 1 / np.e).sum())}/{len(exact)}, "
                      f"mean tuples selected={exact['N_Selected'].mean():.1f}")
                print(f"           max-coverage/OPT mean={exact['MaxCov_OPT_Ratio'].mean():.3f}, "
                      f"min={exact['MaxCov_OPT_Ratio'].min():.3f}, "
                      f"below (1-1/e): {int((exact['MaxCov_OPT_Ratio'] < 1 - 1 / np.e).sum())}/{len(exact)}")
            else:
                print(f"  n={n}, k={k}: no exact OPT; greedy coverage mean={sub_k['Greedy_Coverage'].mean():.3f}, "
                      f"max-coverage mean={sub_k['MaxCov_Coverage'].mean():.3f}, "
                      f"random mean={sub_k['Random_Mean'].mean():.3f}")

    save_results(df, "greedy_efficiency.csv", SEED, "Greedy efficiency")


if __name__ == "__main__":
    run_experiment()
