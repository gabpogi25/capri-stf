"""
experiments/run_invariance.py

invariance experiment — Invariance Verification (QUALITY.md §6.1)

Hypothesis: J1, J2, J3 and Σ1, Σ2, Σ3 are exactly invariant under PGL(4,ℝ),
up to floating-point error.

Null hypothesis: If invariance does NOT hold, relative error grows with
transformation "strength" (cond(A)) rather than sitting flat near machine epsilon.

Parameters:
  - 2,000 random GL(4,ℝ) transforms per configuration
  - 50 six-point configurations (10 generic + 10 boundary + 10 compound +
    10 critical + 10 near-degenerate)
  - Total: 100,000 evaluations
  - Seed: 2042

Success criterion: max relative error < 1e-8 across all trials, AND no
correlation between error and cond(A).
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
    anti_circularity_check, fit_power_law_loglog, bootstrap_ci,
    pick_first_six_tuple, RESULTS_DIR,
)
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual
from algorithms.utils import normalize_to_unit_sphere


EXPERIMENT_ID = 2
SEED = get_experiment_seed(EXPERIMENT_ID)  # 2042
N_TRANSFORMS = 2000
N_CONFIGS_PER_STRATUM = 10
STRATA = ["generic", "boundary", "compound", "critical"]
N_NEAR_DEGENERATE = 10


def generate_random_gl4_matrix(rng: np.random.Generator, max_cond: float = 1e6) -> np.ndarray:
    """
    Sample A ∈ GL(4,ℝ) by drawing 16 i.i.d. N(0,1) entries.
    Reject if |det(A)| < 1e-6 or cond(A) > max_cond.
    """
    for _ in range(1000):
        A = rng.standard_normal((4, 4))
        det_val = np.linalg.det(A)
        if abs(det_val) < 1e-6:
            continue
        cond_val = np.linalg.cond(A)
        if cond_val > max_cond:
            continue
        return A
    # Fallback: return identity-like
    return np.eye(4) + 0.1 * rng.standard_normal((4, 4))


def compute_relative_error(val_before: float, val_after: float) -> float:
    """Relative error with safe denominator to prevent floating-point dust inflation."""
    return abs(val_after - val_before) / max(abs(val_before), 1e-8)


def run_experiment():
    print("=" * 60)
    print("EXP 2: INVARIANCE VERIFICATION")
    print("=" * 60)

    hw_info = log_hardware_info()
    print(f"Hardware: {hw_info.get('processor', 'unknown')}")
    print(f"NumPy: {hw_info.get('numpy_version', 'unknown')}")

    rng = make_rng(SEED)
    results = []

    # Generate 50 configurations: 10 per stratum + 10 near-degenerate
    configs = []
    for stratum in STRATA:
        for i in range(N_CONFIGS_PER_STRATUM):
            pts = generate_stratum_config(stratum, n_points=6, rng=rng)
            configs.append((stratum, pts))

    for i in range(N_NEAR_DEGENERATE):
        kappa = rng.uniform(0.9, 0.99)
        pts = generate_near_degenerate_config(6, kappa=kappa, rng=rng)
        configs.append(("near_degenerate", pts))

    print(f"Generated {len(configs)} configurations")
    print(f"Running {N_TRANSFORMS} transforms per configuration...")

    total_evals = 0
    for config_idx, (stratum, pts_h) in enumerate(configs):
        tuple_idx = pick_first_six_tuple(pts_h.shape[1])

        # Compute invariants BEFORE transform
        inv_before = evaluate_six_point_invariants(pts_h, tuple_idx)
        brackets_before = compute_all_brackets_for_tuple(pts_h, tuple_idx)
        rho_before = compute_syzygy_residual(brackets_before)

        if inv_before is None:
            # Degenerate — J invariants undefined, still check syzygy
            j1_before, j2_before, j3_before = np.nan, np.nan, np.nan
        else:
            j1_before, j2_before, j3_before = inv_before

        for t in range(N_TRANSFORMS):
            A = generate_random_gl4_matrix(rng)
            det_A = float(np.linalg.det(A))
            cond_A = float(np.linalg.cond(A))

            # Apply transform: [x] -> [Ax]
            pts_transformed = A @ pts_h

            # Compute invariants AFTER transform
            inv_after = evaluate_six_point_invariants(pts_transformed, tuple_idx)
            brackets_after = compute_all_brackets_for_tuple(pts_transformed, tuple_idx)
            rho_after = compute_syzygy_residual(brackets_after)

            # Compute relative errors
            rho_error = compute_relative_error(rho_before, rho_after)

            if inv_before is not None and inv_after is not None:
                j1_after, j2_after, j3_after = inv_after
                j1_error = compute_relative_error(j1_before, j1_after)
                j2_error = compute_relative_error(j2_before, j2_after)
                j3_error = compute_relative_error(j3_before, j3_after)
                max_j_error = max(j1_error, j2_error, j3_error)
            else:
                j1_error = j2_error = j3_error = max_j_error = np.nan

            results.append({
                "Config_Index": config_idx,
                "Stratum": stratum,
                "Transform_Index": t,
                "Det_A": det_A,
                "Cond_A": cond_A,
                "J1_RelError": j1_error,
                "J2_RelError": j2_error,
                "J3_RelError": j3_error,
                "Max_J_RelError": max_j_error,
                "Rho_Before": rho_before,
                "Rho_After": rho_after,
                "Rho_RelError": rho_error,
                "Seed": SEED,
            })
            total_evals += 1

        if (config_idx + 1) % 10 == 0:
            print(f"  Config {config_idx + 1}/{len(configs)} done ({total_evals} evaluations)")

    df = pd.DataFrame(results)

    # ---- Summary statistics per stratum ----
    print("\n--- Summary per Stratum ---")
    for stratum in df["Stratum"].unique():
        sub = df[df["Stratum"] == stratum]
        j_errors = sub["Max_J_RelError"].dropna()
        if len(j_errors) > 0:
            print(f"  {stratum}:")
            print(f"    J-invariant max rel error: mean={j_errors.mean():.2e}, "
                  f"median={j_errors.median():.2e}, max={j_errors.max():.2e}, "
                  f"99th={j_errors.quantile(0.99):.2e}")
        rho_errors = sub["Rho_RelError"]
        print(f"    Rho rel error:              mean={rho_errors.mean():.2e}, "
              f"median={rho_errors.median():.2e}, max={rho_errors.max():.2e}")

    # ---- Anti-circularity Check ----
    print("\n--- Anti-Circularity Check ---")
    for stratum in df["Stratum"].unique():
        sub = df[df["Stratum"] == stratum]
        # We must check the variance of the raw values, not the errors (errors are exactly 0).
        raw_rho = sub["Rho_Before"].dropna().values
        if len(raw_rho) > 0:
            anti_circularity_check(raw_rho, label=f"{stratum}/Raw-Rho")

    # ---- Error vs Cond(A) correlation check ----
    print("\n--- Error vs Cond(A) Correlation ---")
    valid = df.dropna(subset=["Max_J_RelError"])
    if len(valid) > 100:
        result = fit_power_law_loglog(valid["Cond_A"].values, valid["Max_J_RelError"].values)
        print(f"  Log-log slope (error vs cond(A)): {result['exponent']:.4f}")
        print(f"  R²: {result['r_squared']:.4f}")
        print(f"  (Should be near-zero slope = no correlation)")

    # ---- Success criterion ----
    max_error = df["Max_J_RelError"].dropna().max()
    print(f"\n--- SUCCESS CRITERION ---")
    print(f"  Max J-invariant relative error: {max_error:.2e}")
    print(f"  Threshold: 1e-8")
    print(f"  PASS: {max_error < 1e-8}")

    save_results(df, "invariance.csv", SEED, "invariance experiment: Invariance Verification")
    print(f"\nTotal evaluations: {total_evals}")


if __name__ == "__main__":
    run_experiment()
