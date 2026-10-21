"""
experiments/run_scaling_law.py

Conditioning-Degeneracy Scaling Law: L(S) ~ d(S)^k

Does the exponent k relating L(S) to distance from a specific stratum
vary by stratum, and does it relate to codim (m-3)?

Parameters:
   - For each stratum (Boundary/Compound/Critical), generate 300 configs
     at controlled nearness kappa in logspace(-5, 0, 20) toward that stratum
     kappa=0 means generic, kappa=1 means exactly degenerate
  - Seed: 42101
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_finding_seed, make_rng, save_results, log_hardware_info,
    generate_near_degenerate_config, anti_circularity_check,
    fit_power_law_loglog, bootstrap_exponent_ci,
    pick_first_six_tuple, RESULTS_DIR,
)
from algorithms.conditioning.scores import compute_conditioning_score


FINDING_ID = 1
SEED = get_finding_seed(FINDING_ID)  # 42101
N_TRIALS = 300
DISTANCE_GRID = np.logspace(-5, 0, 20)


def run_experiment():
    print("=" * 60)
    print("CONDITIONING-DEGENERACY SCALING LAW")
    print("=" * 60)

    log_hardware_info()
    rng = make_rng(SEED)
    results = []

    # For each stratum type, use kappa to control distance.
    # kappa = 1.0 means exactly coplanar (distance = 0).
    # distance ∝ (1 - kappa).
    # Deterministic offsets (avoid hash() which is non-deterministic)
    strata = {
        "Boundary": {"m": 4, "offset": 0},      # m=4, codim=1
        "Compound": {"m": 5, "offset": 10000},   # m=5, codim=2
        "Critical": {"m": 6, "offset": 20000},   # m=6, codim=3
    }

    for stratum_name, info in strata.items():
        m = info["m"]
        stratum_offset = info["offset"]
        codim = m - 3
        print(f"\n  Stratum: {stratum_name} (m={m}, codim={codim})")

        for dist_idx, kappa in enumerate(DISTANCE_GRID):
            # kappa IS the nearness to degeneracy:
            #   kappa ~ 0   -> generic (far from stratum)
            #   kappa ~ 1   -> exactly on the stratum
            # So L(S) should INCREASE with kappa.
            kappa = min(kappa, 0.9999)  # Clamp to avoid exact singularity

            ls_values = []
            for trial in range(N_TRIALS):
                trial_rng = make_rng(SEED + stratum_offset + dist_idx * 1000 + trial)
                # Generate config with m points pulled toward coplanarity by fraction kappa
                n_points = 6
                pts_3d = trial_rng.uniform(-1.0, 1.0, size=(n_points, 3))

                # Pick m points to make near-coplanar
                coplanar_indices = list(trial_rng.choice(n_points, size=min(m, n_points), replace=False))
                normal = trial_rng.standard_normal(3)
                normal /= np.linalg.norm(normal)
                anchor = pts_3d[coplanar_indices[0]]
                plane_d = np.dot(anchor, normal)

                for idx in coplanar_indices:
                    dist_to_plane = np.dot(pts_3d[idx], normal) - plane_d
                    pts_3d[idx] -= kappa * dist_to_plane * normal

                from algorithms.utils import embed_affine_to_homogeneous
                pts_h = embed_affine_to_homogeneous(pts_3d)

                t = pick_first_six_tuple(6)
                ls = compute_conditioning_score(pts_h, t)
                ls_values.append(ls)

                results.append({
                    "Stratum": stratum_name,
                    "Codimension": codim,
                    "Kappa": kappa,
                    "Trial": trial,
                    "L_S": ls,
                    "Seed": SEED,
                })

            mean_ls = np.mean([v for v in ls_values if np.isfinite(v)])
            print(f"    kappa={kappa:.4f}: mean L(S)={mean_ls:.2e}")

    df = pd.DataFrame(results)

    # ---- Per-stratum log-log regression ----
    print("\n--- Per-Stratum Log-Log Regression ---")
    for stratum_name in strata:
        sub = df[df["Stratum"] == stratum_name]
        sub_finite = sub[np.isfinite(sub["L_S"])]
        agg = sub_finite.groupby("Kappa")["L_S"].mean().reset_index()
        x = agg["Kappa"].values
        y = agg["L_S"].values

        result = fit_power_law_loglog(x, y)
        exp_est, ci_lo, ci_hi = bootstrap_exponent_ci(
            sub_finite["Kappa"].values,
            sub_finite["L_S"].values,
            rng=make_rng(SEED + 999),
        )
        print(f"\n  {stratum_name}:")
        print(f"    Fitted exponent k = {result['exponent']:.4f}")
        print(f"    R² = {result['r_squared']:.6f}")
        print(f"    Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ---- Check if exponents are distinct ----
    print("\n--- Are per-stratum exponents distinct? ---")
    print("  (Check if CIs are non-overlapping)")

    save_results(df, "scaling_law.csv", SEED, "Conditioning-degeneracy scaling law")


if __name__ == "__main__":
    run_experiment()
