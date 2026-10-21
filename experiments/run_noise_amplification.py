"""
experiments/run_noise_amplification.py

Stratum-Indexed Noise Amplification Exponent


Hypothesis: Invariant *error* amplifies as noise^codim when noise is injected
near specific degenerate strata, with codim matching Theorem A's (m-3).

What we measure: For each stratum, generate a clean config, compute J1,J2,J3.
Then inject noise, recompute J1,J2,J3, and measure the relative error
  err = max(|J_i_noisy - J_i_clean| / |J_i_clean|)
This is the actual invariant error amplification, NOT the syzygy residual
(which is an algebraic identity and always ~0).

Parameters:
  - Strata: Generic (codim 0, control), Boundary (1), Compound (2), Critical (3)
  - 20 log-spaced noise levels: sigma in [1e-4, 1e-1]
  - 300 trials per (stratum, noise level) cell
  - Seed: 42110

Success criterion: Per-stratum exponents statistically distinguishable and
matching codim (1, 2, 3) with CIs excluding neighboring integers, AND
Generic control showing no such scaling.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    generate_stratum_config, anti_circularity_check,
    fit_power_law_loglog, bootstrap_exponent_ci,
    pick_first_six_tuple, STANDARD_NOISE_GRID, RESULTS_DIR,
)
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.utils import embed_affine_to_homogeneous


FINDING_ID = 10
SEED = get_finding_seed(FINDING_ID)  # 42110
N_TRIALS = 300
NOISE_GRID = STANDARD_NOISE_GRID  # 20 log-spaced from 1e-4 to 1e-1

# Deterministic stratum offsets (avoid hash() which is non-deterministic)
_STRATUM_OFFSETS = {
    "Generic": 0,
    "Boundary": 1000,
    "Compound": 2000,
    "Critical": 3000,
}

STRATA = {
    "Generic": {"codim": 0, "m": 0},
    "Boundary": {"codim": 1, "m": 4},
    "Compound": {"codim": 2, "m": 5},
    "Critical": {"codim": 3, "m": 6},
}


def inject_noise(pts_h: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Inject Gaussian noise into the affine coordinates (rows 1-3)."""
    pts_noisy = pts_h.copy()
    n_points = pts_h.shape[1]
    noise = rng.normal(0, sigma, size=(3, n_points))
    pts_noisy[1:, :] += noise  # rows 1,2,3 are x,y,z (row 0 is the leading 1)
    return pts_noisy


def compute_max_j_relative_error(j_clean, j_noisy):
    """Compute max relative error across J1, J2, J3."""
    errors = []
    for jc, jn in zip(j_clean, j_noisy):
        if abs(jc) < 1e-15:
            errors.append(abs(jn - jc))
        else:
            errors.append(abs(jn - jc) / abs(jc))
    return max(errors)


def run_experiment():
    print("=" * 60)
    print("STRATUM-INDEXED NOISE AMPLIFICATION")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    for stratum_name, stratum_info in STRATA.items():
        codim = stratum_info["codim"]
        stratum_offset = _STRATUM_OFFSETS[stratum_name]
        print(f"\n  Stratum: {stratum_name} (codim={codim})")

        for noise_idx, sigma in enumerate(NOISE_GRID):
            error_values = []

            for trial in range(N_TRIALS):
                trial_rng = make_rng(SEED + stratum_offset + noise_idx * 100 + trial)

                # Generate a CLEAN configuration of the specified stratum type
                pts_clean = generate_stratum_config(
                    stratum_name.lower(), n_points=6, rng=trial_rng
                )

                t = pick_first_six_tuple(6)

                # Compute J1,J2,J3 on the CLEAN points
                j_clean = evaluate_six_point_invariants(pts_clean, t)
                if j_clean is None:
                    # Degenerate: invariants undefined on clean config
                    # Expected for Critical stratum
                    results.append({
                        "Stratum": stratum_name,
                        "Codimension": codim,
                        "Noise_Sigma": sigma,
                        "Trial": trial,
                        "Max_J_Error": np.nan,
                        "J_Clean_Defined": False,
                        "Seed": SEED,
                    })
                    continue

                # Inject noise and recompute
                noise_rng = make_rng(SEED + 90000 + stratum_offset + noise_idx * 100 + trial)
                pts_noisy = inject_noise(pts_clean, sigma, noise_rng)
                j_noisy = evaluate_six_point_invariants(pts_noisy, t)

                if j_noisy is None:
                    max_err = float('inf')
                else:
                    max_err = compute_max_j_relative_error(j_clean, j_noisy)

                error_values.append(max_err)

                results.append({
                    "Stratum": stratum_name,
                    "Codimension": codim,
                    "Noise_Sigma": sigma,
                    "Trial": trial,
                    "Max_J_Error": max_err,
                    "J_Clean_Defined": True,
                    "Seed": SEED,
                })

            # Per-sigma summary
            finite_errors = [e for e in error_values if np.isfinite(e)]
            if finite_errors:
                mean_err = np.mean(finite_errors)
                print(f"    sigma={sigma:.1e}: mean J-error={mean_err:.4e}, "
                      f"n_finite={len(finite_errors)}/{N_TRIALS}")
            else:
                print(f"    sigma={sigma:.1e}: all degenerate or inf")

    df = pd.DataFrame(results)

    # ---- Per-stratum log-log regression ----
    print("\n" + "=" * 60)
    print("LOG-LOG REGRESSION (Invariant Error vs Noise per Stratum)")
    print("=" * 60)

    for stratum_name in STRATA:
        sub = df[(df["Stratum"] == stratum_name) & (df["J_Clean_Defined"] == True)]
        sub_finite = sub[np.isfinite(sub["Max_J_Error"])]
        if len(sub_finite) < 100:
            print(f"\n  {stratum_name}: insufficient finite data ({len(sub_finite)} points)")
            continue

        agg = sub_finite.groupby("Noise_Sigma")["Max_J_Error"].mean().reset_index()
        x = agg["Noise_Sigma"].values
        y = agg["Max_J_Error"].values

        result = fit_power_law_loglog(x, y)
        print(f"\n  {stratum_name} (expected codim={STRATA[stratum_name]['codim']}):")
        print(f"    Fitted exponent k = {result['exponent']:.4f}")
        print(f"    R-squared = {result['r_squared']:.6f}")

        all_x = sub_finite["Noise_Sigma"].values
        all_y = sub_finite["Max_J_Error"].values
        exp_est, ci_lo, ci_hi = bootstrap_exponent_ci(all_x, all_y, rng=make_rng(SEED + 999))
        print(f"    Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

        codim = STRATA[stratum_name]["codim"]
        if codim > 0:
            excludes_below = ci_lo > (codim - 1)
            excludes_above = ci_hi < (codim + 1)
            print(f"    Excludes {codim-1}: {excludes_below}")
            print(f"    Excludes {codim+1}: {excludes_above}")
            print(f"    MATCHES CODIM: {excludes_below and excludes_above}")

    # ---- Anti-circularity summary ----
    print("\n--- Anti-Circularity: Residual Check ---")
    for stratum_name in STRATA:
        sub = df[(df["Stratum"] == stratum_name) & (df["J_Clean_Defined"] == True)]
        sub_finite = sub[np.isfinite(sub["Max_J_Error"])]
        if len(sub_finite) < 100:
            print(f"  {stratum_name}: insufficient data")
            continue
        agg = sub_finite.groupby("Noise_Sigma")["Max_J_Error"].mean().reset_index()
        x = agg["Noise_Sigma"].values
        y = agg["Max_J_Error"].values
        result = fit_power_law_loglog(x, y)
        max_residual = np.max(np.abs(result["residuals"])) if len(result["residuals"]) > 0 else 0
        if max_residual < 1e-10:
            print(f"  [WARNING] {stratum_name}: max log-log residual = {max_residual:.2e} "
                  f"< 1e-10 -- possible formula-in-disguise!")
        else:
            print(f"  [OK] {stratum_name}: max log-log residual = {max_residual:.2e}")

    save_results(df, "noise_amplification.csv", SEED,
                 "Stratum-indexed noise amplification")


if __name__ == "__main__":
    run_experiment()
