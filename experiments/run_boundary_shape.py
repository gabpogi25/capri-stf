"""
experiments/run_boundary_shape.py

Stratum-Transition Boundary Shape

Parameters:
  - 2,000 configs with continuous κ ∈ [0, 1]
  - Fit sigmoid, power law, piecewise-linear; compare via AIC/BIC
  - Seed: 42104
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    generate_near_degenerate_config, anti_circularity_check,
    pick_first_six_tuple, RESULTS_DIR,
)
from algorithms.conditioning.scores import compute_delta_min


FINDING_ID = 4
SEED = get_finding_seed(FINDING_ID)  # 42104
N_TRIALS = 2000


def sigmoid_func(x, a, b, c, d):
    return a * expit(b * (x - c)) + d

def power_law_func(x, a, k):
    return a * np.power(x + 1e-10, k)


def compute_aic(n, k, rss):
    if rss <= 0: rss = 1e-30
    return n * np.log(rss / n) + 2 * k

def compute_bic(n, k, rss):
    if rss <= 0: rss = 1e-30
    return n * np.log(rss / n) + k * np.log(n)


def run_experiment():
    print("=" * 60)
    print("STRATUM-TRANSITION BOUNDARY SHAPE")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    kappa_values = np.linspace(0, 0.999, N_TRIALS)

    for trial, kappa in enumerate(kappa_values):
        trial_rng = make_rng(SEED + trial)
        pts = generate_near_degenerate_config(6, kappa=kappa, rng=trial_rng)
        t = pick_first_six_tuple(6)
        delta_min = compute_delta_min(pts, t)

        results.append({
            "Kappa": kappa,
            "Trial": trial,
            "Delta_Min": delta_min,
            "Seed": SEED,
        })

        if (trial + 1) % 500 == 0:
            print(f"  Trial {trial + 1}/{N_TRIALS}")

    df = pd.DataFrame(results)

    # ---- Fit candidate forms ----
    x = df["Kappa"].values
    y = df["Delta_Min"].values
    n = len(x)

    print("\n--- Model Comparison ---")

    # Sigmoid
    try:
        popt_sig, _ = curve_fit(sigmoid_func, x, y, p0=[0.5, -10, 0.5, 0.0], maxfev=10000)
        y_pred_sig = sigmoid_func(x, *popt_sig)
        rss_sig = np.sum((y - y_pred_sig)**2)
        aic_sig = compute_aic(n, 4, rss_sig)
        bic_sig = compute_bic(n, 4, rss_sig)
        print(f"  Sigmoid:        AIC={aic_sig:.1f}, BIC={bic_sig:.1f}, RSS={rss_sig:.6f}")
    except Exception as e:
        print(f"  Sigmoid: fit failed ({e})")
        aic_sig = bic_sig = np.inf

    # Power law
    try:
        valid = x > 0.001
        popt_pow, _ = curve_fit(power_law_func, x[valid], y[valid], p0=[1.0, -1.0], maxfev=10000)
        y_pred_pow = power_law_func(x[valid], *popt_pow)
        rss_pow = np.sum((y[valid] - y_pred_pow)**2)
        n_pow = np.sum(valid)
        aic_pow = compute_aic(n_pow, 2, rss_pow)
        bic_pow = compute_bic(n_pow, 2, rss_pow)
        print(f"  Power law:      AIC={aic_pow:.1f}, BIC={bic_pow:.1f}, RSS={rss_pow:.6f}")
    except Exception as e:
        print(f"  Power law: fit failed ({e})")
        aic_pow = bic_pow = np.inf

    # Linear
    coeffs_lin = np.polyfit(x, y, 1)
    y_pred_lin = np.polyval(coeffs_lin, x)
    rss_lin = np.sum((y - y_pred_lin)**2)
    aic_lin = compute_aic(n, 2, rss_lin)
    bic_lin = compute_bic(n, 2, rss_lin)
    print(f"  Linear:         AIC={aic_lin:.1f}, BIC={bic_lin:.1f}, RSS={rss_lin:.6f}")

    # Winner
    models = {"Sigmoid": bic_sig, "Power Law": bic_pow, "Linear": bic_lin}
    winner = min(models, key=models.get)
    print(f"\n  Best model (BIC): {winner}")

    save_results(df, "boundary_shape.csv", SEED, "Boundary shape")


if __name__ == "__main__":
    run_experiment()
