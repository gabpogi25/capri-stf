"""
experiments/run_ablation.py

Ablation of the certification gates.

Design
------
1. The fixed threshold L(S) < 1000 accepts 0.8% of generic tuples, so that
   conditioning gate rejects almost everything and hides every other gate.
   Two operating points are reported for L(S) and rho:
     * "manuscript" (data label of the fixed operating point): L(S) < 1000,
       rho <= max(0.05, tau_exact(0.02));
     * "calibrated": thresholds set on a separate calibration sample of
       Generic tuples: L(S) below the generic median (about 50% generic
       acceptance) and rho below the generic 95th percentile.
2. Acceptance is reported separately for GOOD and BAD tuples (TPR / FPR)
   using generator ground truth per tuple, not pooled by regime:
     * Generic, Sparse-View: good;
     * Near-Degenerate: bad if the tuple contains >= 4 of the near-coplanar points;
     * Coplanar: bad if the tuple contains >= 4 of the exactly coplanar points;
     * Mismatched: bad if the tuple contains a replaced point;
     * Non-Rigid (displacement 0.05 > assumed bound 0.02): bad.
3. The shape prior is not part of the certification decision, so there is
   no "no_shape_prior" configuration (it would be identical to the full
   pipeline by construction).
4. rho is computed under the independent-estimation model (two measurements
   with sigma = 1e-3 noise, each bracket from one of them), the same model as
   syzygy-residual experiment, so the consistency gate is tested on data where it can act.
5. Third operating point "pencil" (docs/MATHEMATICS.md, Sections 4-5): the
   conditioning gate uses kappa_J below its generic median, and the
   consistency gate compares the two measurements directly with the pencil
   statistic Q_0.1 |Delta log|CR|| over the 45 pencil cross-ratios of the
   tuple, below its generic 95th percentile. The kappa_J and pencil columns
   use no random numbers, so they do not change the fixed and calibrated rows.

Seed 42009. 1,000 tuples per (regime, configuration, operating point); the same
tuples are used for every configuration so differences are paired.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_experiment_seed, make_rng, save_results, random_six_tuple, bootstrap_ci,
    MEASUREMENT_NOISE_SIGMA,
)
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
    generate_mismatched_configuration,
    generate_non_rigid_configuration,
    generate_sparse_view_configuration,
    measure_with_gaussian_noise,
    estimate_brackets_independently,
)
from algorithms.utils import embed_affine_to_homogeneous
from algorithms.syzygy.residual import compute_syzygy_residual, compute_drift_threshold
from algorithms.conditioning.scores import compute_conditioning_score, compute_invariant_noise_gain
from algorithms.consistency.pencil import pencil_drift_score
from algorithms.degeneracy.stratification import classify_degeneracy_stratum, DegeneracyStratum

EXPERIMENT_ID = 9
SEED = get_experiment_seed(EXPERIMENT_ID)  # 42009
N_TRIALS = 1000
N_CALIBRATION = 2000
N_POINTS = 12
NEAR_DEGENERATE_FRACTION = 0.5          # first 6 of 12 points near one plane
N_EXACT_COPLANAR = 5                    # first 5 points exactly coplanar
ASSUMED_DEFORMATION_BOUND = 0.02
CONFIGS = ["full_pipeline", "no_conditioning", "no_consistency", "no_stratification", "no_drift"]
REGIMES = ["Generic", "Near-Degenerate", "Coplanar", "Mismatched", "Non-Rigid", "Sparse-View"]


def _coplanar_scene(rng: np.random.Generator) -> np.ndarray:
    pts_3d = rng.uniform(-1.0, 1.0, size=(N_POINTS, 3))
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    plane_d = float(np.dot(pts_3d[0], normal))
    for idx in range(N_EXACT_COPLANAR):
        pts_3d[idx] -= (np.dot(pts_3d[idx], normal) - plane_d) * normal
    return embed_affine_to_homogeneous(pts_3d)


def make_trial(regime: str, rng: np.random.Generator) -> dict:
    """Scene, two measurements, tuple and ground-truth label for one trial."""
    n_points = 8 if regime == "Sparse-View" else N_POINTS
    t = random_six_tuple(n_points, rng)
    bad = False
    if regime == "Generic":
        scene_a = scene_b = generate_generic_configuration(n_points, scale=1.0, rng=rng)
    elif regime == "Near-Degenerate":
        scene_a = scene_b = generate_near_degenerate_configuration(
            n_points, coplanar_fraction=NEAR_DEGENERATE_FRACTION, plane_offset=1e-6, scale=1.0, rng=rng)
        bad = len(set(t) & set(range(int(n_points * NEAR_DEGENERATE_FRACTION)))) >= 4
    elif regime == "Coplanar":
        scene_a = scene_b = _coplanar_scene(rng)
        bad = len(set(t) & set(range(N_EXACT_COPLANAR))) >= 4
    elif regime == "Mismatched":
        scene_a, scene_b, corrupted = generate_mismatched_configuration(
            n_points, mismatch_fraction=0.15, scale=1.0, rng=rng)
        bad = bool(set(t) & {int(i) for i in corrupted})
    elif regime == "Non-Rigid":
        scene_a, scene_b, _ = generate_non_rigid_configuration(
            n_points, deformation_bound=0.05, scale=1.0, rng=rng)
        bad = True
    elif regime == "Sparse-View":
        scene_a, _ = generate_sparse_view_configuration(
            n_points, n_views=2, points_per_view=6, scale=1.0, rng=rng)
        scene_b = scene_a
    else:
        raise ValueError(regime)
    measurements = [measure_with_gaussian_noise(scene_a, MEASUREMENT_NOISE_SIGMA, rng),
                    measure_with_gaussian_noise(scene_b, MEASUREMENT_NOISE_SIGMA, rng)]
    brackets, _ = estimate_brackets_independently(measurements, t, rng)
    return {
        "t": t,
        "bad": bad,
        "rho": compute_syzygy_residual(brackets),
        "ls": compute_conditioning_score(scene_a, t),
        "kappa": compute_invariant_noise_gain(scene_a, t),
        "pencil": pencil_drift_score(measurements[0][:, list(t)], measurements[1][:, list(t)]),
        "stratum": classify_degeneracy_stratum(scene_a, t),
    }


def decide(trial: dict, config: str, ls_threshold: float, rho_threshold: float, tau: float,
           conditioning_key: str = "ls", consistency_key: str = "rho") -> bool:
    """Apply the gates that are switched on for this configuration."""
    stratum_ok = trial["stratum"] not in {DegeneracyStratum.CRITICAL, DegeneracyStratum.COMPOUND}
    conditioning_ok = trial[conditioning_key] < ls_threshold
    consistency_limit = rho_threshold if config == "no_drift" else max(rho_threshold, tau)
    consistency_ok = trial[consistency_key] <= consistency_limit
    gates = {
        "stratification": stratum_ok,
        "conditioning": conditioning_ok,
        "consistency": consistency_ok,
    }
    if config == "no_conditioning":
        gates.pop("conditioning")
    elif config == "no_consistency":
        gates.pop("consistency")
    elif config == "no_stratification":
        gates.pop("stratification")
    return all(gates.values())


def calibrate_thresholds() -> dict[str, float]:
    """Generic-only calibration sample, disjoint seeds from the test trials."""
    results = [make_trial("Generic", make_rng(SEED + 900000 + trial)) for trial in range(N_CALIBRATION)]
    values = {key: np.asarray([r[key] for r in results], dtype=float) for key in ("ls", "rho", "kappa", "pencil")}
    finite = {key: v[np.isfinite(v)] for key, v in values.items()}
    return {
        "ls": float(np.median(finite["ls"])),
        "rho": float(np.quantile(finite["rho"], 0.95)),
        "kappa": float(np.median(finite["kappa"])),
        "pencil": float(np.quantile(finite["pencil"], 0.95)),
    }


def run_experiment() -> pd.DataFrame:
    print("=" * 60)
    print("EXP 9: ABLATION AT THREE OPERATING POINTS")
    print("=" * 60)
    calibrated = calibrate_thresholds()
    tau = compute_drift_threshold(ASSUMED_DEFORMATION_BOUND)
    # name -> (conditioning threshold, consistency threshold, tau, conditioning key, consistency key)
    operating_points = {
        "manuscript": (1000.0, 0.05, tau, "ls", "rho"),
        "calibrated": (calibrated["ls"], calibrated["rho"], 0.0, "ls", "rho"),
        "pencil": (calibrated["kappa"], calibrated["pencil"], 0.0, "kappa", "pencil"),
    }
    print(f"  Calibrated thresholds: L(S) < {calibrated['ls']:.1f}, rho <= {calibrated['rho']:.3e}")
    print(f"  Pencil thresholds:     kappa_J < {calibrated['kappa']:.2f}, pencil Q0.1 <= {calibrated['pencil']:.3e}")

    trials = {
        regime: [make_trial(regime, make_rng(SEED + regime_index * 10000 + k)) for k in range(N_TRIALS)]
        for regime_index, regime in enumerate(REGIMES)
    }
    rows = []
    for op_name, (ls_threshold, rho_threshold, op_tau, cond_key, cons_key) in operating_points.items():
        for config in CONFIGS:
            for regime in REGIMES:
                for k, trial in enumerate(trials[regime]):
                    rows.append({
                        "Operating_Point": op_name,
                        "Configuration": config,
                        "Regime": regime,
                        "Trial": k,
                        "Tuple_Bad": trial["bad"],
                        "Accepted": decide(trial, config, ls_threshold, rho_threshold, op_tau, cond_key, cons_key),
                        "Rho": trial["rho"],
                        "LS": trial["ls"],
                        "Kappa": trial["kappa"],
                        "Pencil_Q10": trial["pencil"],
                        "Conditioning_Metric": cond_key,
                        "Consistency_Metric": cons_key,
                        "Stratum": trial["stratum"].value,
                        "LS_Threshold": ls_threshold,
                        "Rho_Threshold": max(rho_threshold, op_tau) if config != "no_drift" else rho_threshold,
                        "Seed": SEED,
                    })
    df = pd.DataFrame(rows)

    print("\nTPR (good tuples accepted) / FPR (bad tuples accepted), 95% bootstrap CI:")
    for (op_name, config), group in df.groupby(["Operating_Point", "Configuration"], sort=False):
        good = group.loc[~group["Tuple_Bad"], "Accepted"].astype(float).to_numpy()
        bad = group.loc[group["Tuple_Bad"], "Accepted"].astype(float).to_numpy()
        tpr, tpr_lo, tpr_hi = bootstrap_ci(good, rng=make_rng(SEED + 1))
        fpr, fpr_lo, fpr_hi = bootstrap_ci(bad, rng=make_rng(SEED + 2))
        print(f"  {op_name:10s} {config:18s} TPR={tpr:.3f} [{tpr_lo:.3f},{tpr_hi:.3f}]  "
              f"FPR={fpr:.3f} [{fpr_lo:.3f},{fpr_hi:.3f}]  (n_good={len(good)}, n_bad={len(bad)})")

    save_results(df, "ablation.csv", SEED, "ablation study: Ablation at fixed (L < 1000), calibrated and pencil operating points")
    return df


if __name__ == "__main__":
    run_experiment()
