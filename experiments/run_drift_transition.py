"""
experiments/run_drift_transition.py

Critical Drift Ratio (Phase-Transition Point)

Extends drift-admission experiment with normalized η/L(S) ratio analysis and formal
changepoint detection.

Generator: two measurements whose unit representatives move by at most eta
(Theorem 6.1 model), each bracket estimated from one of them. Theorem 6.1
proves rho <= tau_exact(eta), so admission is 100% at every eta and there is
no transition to detect; the experiment documents that, and Rho records how
far below the bound the residual sits.

Parameters:
  - Same grid as drift-admission experiment (η ∈ [0.001, 0.2])
  - Additionally sweep ratio η / L(S)^(-1/2)
  - Formal changepoint detection via second-derivative maximum
  - Seed: 42102
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    anti_circularity_check, bootstrap_ci, random_six_tuple, RESULTS_DIR,
)
from algorithms.syzygy.residual import compute_syzygy_residual, compute_drift_threshold
from algorithms.conditioning.scores import compute_conditioning_score
from synthetic_data.generator import (
    generate_generic_configuration,
    perturb_within_ball,
    estimate_brackets_independently,
)


FINDING_ID = 2
SEED = get_finding_seed(FINDING_ID)  # 42102
N_TRIALS = 300
N_POINTS = 12


def detect_changepoint_second_derivative(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Detect changepoint via second-derivative maximum.
    Returns changepoint location and its properties.
    """
    if len(x) < 5:
        return {"changepoint_x": np.nan, "changepoint_idx": -1, "max_second_deriv": 0.0}

    # Compute second derivative via finite differences
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)

    # Find maximum absolute second derivative
    max_idx = np.argmax(np.abs(d2y))

    return {
        "changepoint_x": float(x[max_idx]),
        "changepoint_idx": int(max_idx),
        "max_second_deriv": float(d2y[max_idx]),
        "second_derivative": d2y,
    }


def run_experiment():
    print("=" * 60)
    print("CRITICAL DRIFT RATIO TRANSITION")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)

    etas = np.logspace(np.log10(0.001), np.log10(0.2), 50)
    results = []

    for eta_idx, eta in enumerate(etas):
        for trial in range(N_TRIALS):
            trial_rng = make_rng(SEED + eta_idx * 10000 + trial)

            # Two bounded-drift measurements (Theorem 6.1 model), each bracket
            # estimated from one of them (replaces bracket averaging).
            pts = generate_generic_configuration(N_POINTS, scale=1.0, rng=trial_rng)
            t = random_six_tuple(N_POINTS, trial_rng)
            measurements = [perturb_within_ball(pts, eta, trial_rng),
                            perturb_within_ball(pts, eta, trial_rng)]
            brackets, _ = estimate_brackets_independently(measurements, t, trial_rng)
            rho = compute_syzygy_residual(brackets)
            ls = compute_conditioning_score(pts, t)
            tau = compute_drift_threshold(eta)
            admitted = rho <= tau

            # Normalized ratio: η / L(S)^(-1/2) = η * sqrt(L(S))
            ls_safe = max(ls, 1e-12)
            normalized_ratio = eta * np.sqrt(ls_safe)

            results.append({
                "Eta": eta,
                "Trial": trial,
                "Rho": rho,
                "LS": ls,
                "Tau": tau,
                "Admitted": admitted,
                "Normalized_Ratio": normalized_ratio,
                "Seed": SEED,
            })

        # Progress
        if (eta_idx + 1) % 10 == 0:
            print(f"  eta step {eta_idx + 1}/50 done")

    df = pd.DataFrame(results)

    # ---- Accuracy vs raw η ----
    print("\n--- Changepoint Analysis: Raw eta ---")
    agg_raw = df.groupby("Eta")["Admitted"].mean().reset_index()
    cp_raw = detect_changepoint_second_derivative(
        agg_raw["Eta"].values, agg_raw["Admitted"].values
    )
    print(f"  Changepoint (raw eta): {cp_raw['changepoint_x']:.4f}")
    print(f"  Max |d²/deta²|: {cp_raw['max_second_deriv']:.4f}")

    # ---- Accuracy vs normalized ratio ----
    print("\n--- Changepoint Analysis: Normalized Ratio ---")
    df["Ratio_Bin"] = pd.cut(df["Normalized_Ratio"], bins=50, labels=False)
    agg_norm = df.groupby("Ratio_Bin").agg(
        Ratio_Mean=("Normalized_Ratio", "mean"),
        Admitted_Mean=("Admitted", "mean"),
    ).dropna().reset_index()
    cp_norm = detect_changepoint_second_derivative(
        agg_norm["Ratio_Mean"].values, agg_norm["Admitted_Mean"].values
    )
    print(f"  Changepoint (normalized): {cp_norm['changepoint_x']:.4f}")
    print(f"  Max |d²/deta²|: {cp_norm['max_second_deriv']:.4f}")

    # ---- Bootstrap CI on changepoint ----
    print("\n--- Bootstrap CI on Changepoint Location ---")
    cp_locations = []
    for boot in range(500):
        boot_rng = make_rng(SEED + 50000 + boot)
        sample_idx = boot_rng.choice(len(agg_raw), size=len(agg_raw), replace=True)
        boot_x = agg_raw["Eta"].values[sample_idx]
        boot_y = agg_raw["Admitted"].values[sample_idx]
        sort_order = np.argsort(boot_x)
        boot_x = boot_x[sort_order]
        boot_y = boot_y[sort_order]
        cp = detect_changepoint_second_derivative(boot_x, boot_y)
        if np.isfinite(cp["changepoint_x"]):
            cp_locations.append(cp["changepoint_x"])

    if len(cp_locations) > 50:
        cp_arr = np.array(cp_locations)
        ci_lo = float(np.percentile(cp_arr, 2.5))
        ci_hi = float(np.percentile(cp_arr, 97.5))
        print(f"  Changepoint 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
        ci_width = ci_hi - ci_lo
        eta_range = 0.2 - 0.001
        print(f"  CI width / total range: {ci_width/eta_range:.3f}")
        if ci_width > eta_range * 0.8:
            print(f"  [WARNING] CI spans >80% of range -- no sharp transition detected")
    else:
        print(f"  Could not compute bootstrap CI (too few valid samples)")

    # ---- Anti-circularity ----
    print("\n--- Anti-Circularity ---")
    for eta in etas[::10]:
        sub = df[df["Eta"] == eta]
        anti_circularity_check(sub["Rho"].values, label=f"eta={eta:.4f}/rho")

    save_results(df, "drift_transition.csv", SEED, "Critical drift ratio")


if __name__ == "__main__":
    run_experiment()
