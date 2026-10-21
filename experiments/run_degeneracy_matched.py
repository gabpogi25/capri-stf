"""
experiments/run_degeneracy_matched.py

matched degeneracy experiment: degeneracy detection of 8-point samples with matched inputs (pre-registration
docs/prereg/degeneracy_matched.md, which governs every choice below; negative finding N5,
GOAL row 17). Synthetic data only.

Label: a sample is degenerate iff >= 6 of its 8 points lie on the dominant plane
(secondary: >= 7). Scene generator, baseline scorers and constants are imported
read-only from run_synthetic_baselines.py.

Arm M8 (primary): every method sees only the 8 correspondences (and sigma_px).
CAPRI reconstructs from those 8 alone (algorithms/degeneracy/coplanarity.py: linear
8-point F in the Hartley charts of the 8 points, canonical cameras, DLT). capri_kappa
and capri_sigma are therefore evaluated in the 8-point Hartley reconstruction chart,
where pixel noise is not isotropic, so both are chart-dependent and the noise reading
of kappa_J holds only qualitatively.
Arm SC (secondary): every method may use the whole scene (USAC-MAGSAC F of
projective_reconstruction with fixed cameras for CAPRI; USAC-MAGSAC scene homography
for the DEGENSAC-style counts and for GRIC).

Seeds (S = get_experiment_seed(18) = 42018): calibration trials S + 60,000,000 + t
(t < 20), test trials S + 61,000,000 + t (t < 80), bootstrap S + 62,000,000. One
make_rng(seed) per trial: scene first, then the 108 samples; cv2.setRNGSeed(seed)
before each scene-level USAC call. The sample size is fixed and does not depend on
runtime. Trials may run in worker processes (ordered map); results are assembled in
trial order and do not depend on the number of workers. Runtime goes to stdout only.

Outputs (save_results, seed S): degeneracy_matched_samples.csv, degeneracy_matched_summary.csv,
degeneracy_matched_comparisons.csv, degeneracy_matched_by_planar_count.csv, degeneracy_matched_stratum.csv.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_synthetic_baselines as rsb
from experiment_utils import get_experiment_seed, make_rng, save_results
from algorithms.conditioning.scores import batch_invariant_noise_gain
from algorithms.degeneracy import coplanarity as cop
from reconstruction.triangulation import estimate_projection_matrices_from_fundamental

EXPERIMENT_SEED = get_experiment_seed(18)  # 42018
SPLITS = {"calibration": (60_000_000, 20), "test": (61_000_000, 80)}
BOOTSTRAP_SEED = EXPERIMENT_SEED + 62_000_000
N_BOOTSTRAP, SAMPLES_PER_K, N_SAMPLE_POINTS = 2000, 12, 8
STEP_PX, CHECK_STEP_PX, STEP_AGREEMENT_RTOL = 1e-3, 1e-4, 1e-3
SUBSET_SIZES = (5, 6, 7, 8)
DLT_RANK_RELATIVE_TOL = 1e-12
MOST, LEAST, IMPORTED_INFINITY_MARKER = 1e300, -1e300, 1e300
TARGET_FPR = 0.05
SUBSETS_4, SUBSETS_6 = cop.subsets_of_size(8, 4), cop.subsets_of_size(8, 6)
METHODS = {
    "M8": ["capri_coplanarity", "capri_sigma", "capri_kappa", "capri_ls", "degensac_h_inliers", "gric_h_minus_f",
           "svd_condition", "degensac_e6", "homography_ls6", "neg_delta_6"],
    "SC": ["capri_coplanarity", "capri_sigma", "capri_kappa", "capri_ls", "degensac_scene_count",
           "degensac_scene_e6", "gric_scene", "svd_condition", "neg_delta_6"]}
WITH_OPPOSITE_VARIANT = {"capri_coplanarity", "capri_kappa", "degensac_e6", "homography_ls6", "degensac_scene_e6"}
RULES = ("primary", "opposite")


# --------------------------------------------------------------------------- 2-D scores

def homography_subset_errors(view_1: np.ndarray, view_2: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """Transfer errors (S, 8) of all 8 points under the DLT homography of each subset
    (fit_homography_dlt and transfer_errors of run_synthetic_baselines, unchanged)."""
    return np.array([rsb.transfer_errors(rsb.fit_homography_dlt(view_1[s], view_2[s]), view_1, view_2) for s in subsets])


def count_rank_deficient_dlt(view_1: np.ndarray, view_2: np.ndarray, subsets: np.ndarray) -> int:
    """Number of subsets whose Hartley-normalised DLT matrix has s_8/s_1 < 1e-12 (counted only)."""
    rows = []
    for subset in subsets:
        src, dst = rsb.hartley_normalize(view_1[subset])[0], rsb.hartley_normalize(view_2[subset])[0]
        ones, zeros = np.ones(len(subset)), np.zeros((len(subset), 3))
        source = np.column_stack([src, ones])
        rows.append(np.vstack([np.hstack([-source, zeros, dst[:, :1] * source]),
                               np.hstack([zeros, -source, dst[:, 1:] * source])]))
    singular = np.linalg.svd(np.array(rows), compute_uv=False)
    return int(np.sum(singular[:, 7] < DLT_RANK_RELATIVE_TOL * singular[:, 0]))


def sixth_smallest_error_score(errors: np.ndarray, infinite_error: float = np.inf) -> float:
    """degensac_e6 = -min over rows of the 6th-smallest transfer error (higher = more degenerate)."""
    assert errors.ndim == 2 and errors.shape[1] >= 6
    return float(-np.min(np.sort(np.where(np.isinf(errors), infinite_error, errors), axis=1)[:, 5]))


def subset_rms_error_score(errors: np.ndarray, subsets: np.ndarray, infinite_error: float = np.inf) -> float:
    """homography_ls6 = -min over subsets of the RMS transfer error of the subset's own points."""
    assert errors.shape[0] == subsets.shape[0]
    own = np.take_along_axis(np.where(np.isinf(errors), infinite_error, errors), subsets, axis=1)
    return float(-np.min(np.sqrt(np.mean(own ** 2, axis=1))))


def gric_difference(view_1: np.ndarray, view_2: np.ndarray, fundamental_px: np.ndarray, homography_px: np.ndarray) -> float:
    """GRIC(F) - GRIC(H) with given (scene) models; same Torr constants and residuals as
    rsb.score_gric, which fits its own models from the sample and so cannot be reused here."""
    n, sigma2, r = len(view_1), rsb.NOISE_PX ** 2, 4
    lambda_1, lambda_2, lambda_3 = np.log(r), np.log(r * n), 2.0
    x1, x2 = np.column_stack([view_1, np.ones(n)]), np.column_stack([view_2, np.ones(n)])
    fx1, ftx2 = x1 @ fundamental_px.T, x2 @ fundamental_px
    f_residuals = np.sum(x2 * fx1, axis=1) ** 2 / np.maximum(fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2
                                                             + ftx2[:, 1] ** 2, 1e-300)
    h_residuals = rsb.transfer_errors(homography_px, view_1, view_2) ** 2

    def gric(residuals_sq: np.ndarray, d: int, k: int) -> float:
        return float(np.sum(np.minimum(residuals_sq / sigma2, lambda_3 * (r - d))) + lambda_1 * d * n + lambda_2 * k)

    return gric(f_residuals, 3, 7) - gric(h_residuals, 2, 8)


# --------------------------------------------------------------------------- CAPRI scores

def nan_safe_min(values: np.ndarray) -> float:
    """Minimum over non-NaN entries; NaN if all are NaN."""
    return float("nan") if np.all(np.isnan(values)) else float(np.nanmin(values))


def coplanarity_columns(prefix: str, reconstruct, view_1: np.ndarray) -> tuple[dict, np.ndarray]:
    """capri_coplanarity (both rules), m_hat, T of the first six points, both step sizes and counts,
    and the unperturbed reconstruction (4, 8). `reconstruct(step)` returns the reconstructions
    (B, 4, 8) of the perturbed copies; a failed SVD makes every subset non-finite."""
    try:
        perturbed = reconstruct(STEP_PX)
        profile = cop.stratum_distance_profile(perturbed, view_1, SUBSET_SIZES, STEP_PX, rsb.NOISE_PX)
        check, _ = cop.stratum_distance_profile(reconstruct(CHECK_STEP_PX), view_1, (6,), CHECK_STEP_PX, rsb.NOISE_PX)[6]
        unperturbed = perturbed[0]
    except np.linalg.LinAlgError:
        nan_profile = (np.full(len(SUBSETS_6), np.nan), np.full(len(SUBSETS_6), cop.STATUS_NONFINITE))
        profile, check = {m: nan_profile for m in SUBSET_SIZES}, nan_profile[0]
        unperturbed = np.full((4, N_SAMPLE_POINTS), np.nan)
    distances, status = profile[6]
    counts = np.bincount(status, minlength=4)
    return {f"{prefix}capri_coplanarity": cop.coplanarity_score(distances, status),
            f"{prefix}capri_coplanarity_opp": cop.coplanarity_score(distances, status, infeasible_distance=0.0),
            f"{prefix}m_hat": cop.estimate_stratum_size({m: nan_safe_min(profile[m][0]) for m in SUBSET_SIZES}),
            f"{prefix}T_first_six": distances[0], f"{prefix}min_T6": nan_safe_min(distances),
            f"{prefix}min_T6_step_1e4": nan_safe_min(check),
            f"{prefix}n_nonfinite_subsets": counts[cop.STATUS_NONFINITE],
            f"{prefix}n_rankdef_feasible": counts[cop.STATUS_FEASIBLE_PINV],
            f"{prefix}n_rankdef_infeasible": counts[cop.STATUS_INFEASIBLE]}, unperturbed


def conditioning_columns(prefix: str, points_3d: np.ndarray, sample: np.ndarray) -> dict:
    """capri_ls, capri_sigma, capri_kappa via score_capri / score_capri_kappa (unchanged); the imported
    1e300 marker for a non-finite subset is read back as +inf (mapped to MOST). Opposite rule for kappa:
    NaN subsets left out of the max. delta_m profile of the unit columns."""
    if not np.all(np.isfinite(points_3d[:, sample])):
        return {f"{prefix}{name}": np.nan for name in ("capri_ls", "capri_sigma", "capri_kappa", "capri_kappa_opp")}
    (capri_ls, capri_sigma), capri_kappa = rsb.score_capri(points_3d, sample), rsb.score_capri_kappa(points_3d, sample)
    raw_kappa = batch_invariant_noise_gain(points_3d, sample[SUBSETS_6])
    columns = {f"{prefix}capri_ls": np.inf if capri_ls >= IMPORTED_INFINITY_MARKER else capri_ls,
               f"{prefix}capri_kappa": np.inf if capri_kappa >= IMPORTED_INFINITY_MARKER else capri_kappa,
               f"{prefix}capri_sigma": capri_sigma, f"{prefix}capri_kappa_opp": -nan_safe_min(-raw_kappa)}
    for subset_size in SUBSET_SIZES:
        columns[f"{prefix}delta_{subset_size}"] = cop.coplanarity_gap(points_3d[:, sample], subset_size)
    columns[f"{prefix}neg_delta_6"] = -columns[f"{prefix}delta_6"]
    return columns


def score_matched_sample(view_1: np.ndarray, view_2: np.ndarray) -> dict:
    """Arm M8: every score from the 8 correspondences alone."""
    chart_1, chart_2 = rsb.hartley_normalize(view_1)[1], rsb.hartley_normalize(view_2)[1]

    def reconstruct(step: float) -> np.ndarray:
        return cop.batch_eight_point_reconstruction(*cop.perturbed_correspondences(view_1, view_2, step), chart_1, chart_2)[0]

    columns, unperturbed = coplanarity_columns("m8_", reconstruct, view_1)
    columns.update(conditioning_columns("m8_", unperturbed, np.arange(N_SAMPLE_POINTS)))
    errors_4, errors_6 = homography_subset_errors(view_1, view_2, SUBSETS_4), homography_subset_errors(view_1, view_2, SUBSETS_6)
    columns.update({
        "m8_degensac_h_inliers": rsb.score_degensac(view_1, view_2), "m8_gric_h_minus_f": rsb.score_gric(view_1, view_2),
        "m8_svd_condition": rsb.score_svd(view_1, view_2),
        "m8_degensac_e6": sixth_smallest_error_score(errors_4), "m8_degensac_e6_opp": sixth_smallest_error_score(errors_4, 0.0),
        "m8_homography_ls6": subset_rms_error_score(errors_6, SUBSETS_6),
        "m8_homography_ls6_opp": subset_rms_error_score(errors_6, SUBSETS_6, 0.0),
        "m8_n_rankdef_dlt_e6": count_rank_deficient_dlt(view_1, view_2, SUBSETS_4),
        "m8_n_rankdef_dlt_ls6": count_rank_deficient_dlt(view_1, view_2, SUBSETS_6),
        "m8_n_inf_transfer_e6": int(np.sum(np.isinf(errors_4))), "m8_n_inf_transfer_ls6": int(np.sum(np.isinf(errors_6)))})
    return columns


def scene_models(view_1: np.ndarray, view_2: np.ndarray, seed: int) -> dict:
    """Scene reconstruction (projective_reconstruction, unchanged), its cameras and charts (same
    USAC-MAGSAC call and seed, recomputed because projective_reconstruction returns points only),
    and the scene homography (USAC-MAGSAC, 2 px). A failed homography gives NaN entries, whose
    transfer errors are +inf (prereg 6.1)."""
    points_3d = rsb.projective_reconstruction(view_1, view_2, seed)
    (norm_1, chart_1), (norm_2, chart_2) = rsb.hartley_normalize(view_1), rsb.hartley_normalize(view_2)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(view_1 - view_1.mean(axis=0), axis=1))
    cv2.setRNGSeed(seed)
    fundamental, _ = cv2.findFundamentalMat(norm_1, norm_2, cv2.USAC_MAGSAC, 1.0 * scale, 0.999)
    camera_1, camera_2 = estimate_projection_matrices_from_fundamental(fundamental[:3], norm_1, norm_2)
    cv2.setRNGSeed(seed)
    homography, _ = cv2.findHomography(view_1, view_2, cv2.USAC_MAGSAC, rsb.H_INLIER_PX, maxIters=20000, confidence=0.9999)
    homography = np.full((3, 3), np.nan) if homography is None else homography
    return {"points_3d": points_3d, "chart_1": chart_1, "chart_2": chart_2, "camera_1": camera_1, "camera_2": camera_2,
            "fundamental_px": chart_2.T @ fundamental[:3] @ chart_1, "homography_px": homography}


def score_scene_sample(view_1: np.ndarray, view_2: np.ndarray, sample: np.ndarray, scene: dict) -> dict:
    """Arm SC: every method may use the scene-level F, cameras and homography."""
    def reconstruct(step: float) -> np.ndarray:
        return cop.batch_fixed_camera_triangulation(*cop.perturbed_correspondences(view_1, view_2, step), scene["chart_1"],
                                                    scene["chart_2"], scene["camera_1"], scene["camera_2"])

    columns, unperturbed = coplanarity_columns("sc_", reconstruct, view_1)
    scene_unit = scene["points_3d"][:, sample] / np.linalg.norm(scene["points_3d"][:, sample], axis=0)
    assert not np.all(np.isfinite(unperturbed)) or np.all(np.abs(np.sum(scene_unit * unperturbed, axis=0)) > 1 - 1e-9)
    columns.update(conditioning_columns("sc_", scene["points_3d"], sample))
    errors = rsb.transfer_errors(scene["homography_px"], view_1, view_2)
    columns.update({
        "sc_degensac_scene_count": float(np.sum(errors < rsb.H_INLIER_PX)), "sc_degensac_scene_e6": sixth_smallest_error_score(errors[None]),
        "sc_degensac_scene_e6_opp": sixth_smallest_error_score(errors[None], 0.0),
        "sc_gric_scene": gric_difference(view_1, view_2, scene["fundamental_px"], scene["homography_px"]),
        "sc_n_inf_transfer_scene": int(np.sum(np.isinf(errors)))})
    return columns


def run_trial(task: tuple[str, int, int]) -> list[dict]:
    """All samples of one trial; depends only on the trial seed (safe to run in any process)."""
    split, trial, samples_per_k = task
    seed = EXPERIMENT_SEED + SPLITS[split][0] + trial
    cv2.setNumThreads(1)
    rng = make_rng(seed)
    view_1, view_2, planar = rsb.make_scene(rng)
    scene = scene_models(view_1, view_2, seed)
    planar_idx, generic_idx = np.flatnonzero(planar), np.flatnonzero(~planar)
    rows = []
    for k in range(N_SAMPLE_POINTS + 1):
        for _ in range(samples_per_k):
            sample = np.r_[rng.choice(planar_idx, k, replace=False), rng.choice(generic_idx, N_SAMPLE_POINTS - k, replace=False)]
            row = {"Split": split, "Trial": trial, "Scene_Seed": seed, "N_Planar": k,
                   "Degenerate": int(k >= rsb.DEGENERATE_MIN_PLANAR), "Degenerate_k7": int(k >= 7)}
            row.update({f"Point_{i + 1}": int(index) for i, index in enumerate(sample)})
            row.update(score_matched_sample(view_1[sample], view_2[sample]))
            row.update(score_scene_sample(view_1[sample], view_2[sample], sample, scene))
            row["sc_svd_condition"] = row["m8_svd_condition"]   # no scene version (prereg Section 4)
            rows.append(row)
    return rows


def collect_samples(n_workers: int) -> pd.DataFrame:
    """Every trial of both splits, assembled in (split, trial) order."""
    tasks = [(split, trial, SAMPLES_PER_K) for split, (_, n_trials) in SPLITS.items() for trial in range(n_trials)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return pd.DataFrame([row for rows in pool.map(run_trial, tasks) for row in rows])


# --------------------------------------------------------------------------- statistics

def map_nonfinite(values: np.ndarray, rule: str) -> np.ndarray:
    """Sample-level rule (prereg 6.1): +inf -> MOST, -inf -> LEAST, NaN -> LEAST (primary) / MOST (opposite)."""
    assert rule in RULES
    finite = np.isfinite(values)
    assert np.all(np.abs(values[finite]) < 1e299), "finite score collides with a sentinel"
    mapped = np.where(values == np.inf, MOST, np.where(values == -np.inf, LEAST, values))
    return np.where(np.isnan(values), LEAST if rule == "primary" else MOST, mapped)


def score_column(arm: str, method: str, rule: str) -> str:
    """Column of the raw score for (arm, method, rule)."""
    suffix = "_opp" if rule == "opposite" and method in WITH_OPPOSITE_VARIANT else ""
    return f"{arm.lower()}_{method}{suffix}"


def weighted_auc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUC (ties count 1/2) with per-sample weights (B, n), one AUC per row."""
    order = np.argsort(scores, kind="mergesort")
    ordered, positive = scores[order], labels[order] == 1
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    w = weights[:, order]
    pos, neg = np.add.reduceat(w * positive, starts, axis=1), np.add.reduceat(w * ~positive, starts, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return np.sum(pos * (below + 0.5 * neg), axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def calibration_threshold(negative_scores: np.ndarray) -> float:
    """Pre-registered tie rule: thr = sorted negatives[ceil(0.95 n) - 1] (as rsb.tpr_at_fpr); flag iff score > thr."""
    ordered = np.sort(negative_scores)
    return float(ordered[int(np.ceil((1 - TARGET_FPR) * len(ordered))) - 1])


def percentile_interval(draws: np.ndarray, level: float) -> tuple[float, float]:
    """Percentile bootstrap interval at the given two-sided level."""
    return float(np.percentile(draws, 50 * (1 - level))), float(np.percentile(draws, 50 * (1 + level)))


def bootstrap_sample_weights(test: pd.DataFrame) -> np.ndarray:
    """One set of N_BOOTSTRAP resampled trial sets (seed S + 62,000,000), as per-sample weights (B, n)."""
    n_trials = SPLITS["test"][1]
    draws = make_rng(BOOTSTRAP_SEED).integers(0, n_trials, size=(N_BOOTSTRAP, n_trials))
    counts = np.stack([np.bincount(draw, minlength=n_trials) for draw in draws]).astype(float)
    return counts[:, test["Trial"].to_numpy()]


SUBSET_COUNT_COLUMNS = {"capri_coplanarity": ("n_nonfinite_subsets", "n_rankdef_feasible", "n_rankdef_infeasible"),
                        "degensac_e6": ("n_rankdef_dlt_e6", "n_inf_transfer_e6"),
                        "homography_ls6": ("n_rankdef_dlt_ls6", "n_inf_transfer_ls6"),
                        **{m: ("n_inf_transfer_scene",) for m in ("degensac_scene_count", "degensac_scene_e6", "gric_scene")}}


def mapping_counts(raw: np.ndarray, labels: np.ndarray, split: str) -> dict:
    """Samples mapped from +inf, -inf and NaN, by label class (prereg 6.1 counts)."""
    return {f"N_{kind}_{split}_label{c}": int(np.sum(test(raw[labels == c])))
            for kind, test in (("posinf", np.isposinf), ("neginf", np.isneginf), ("nan", np.isnan)) for c in (0, 1)}


def score_summary(test: pd.DataFrame, calibration: pd.DataFrame, weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """AUC (pooled, bootstrap CI, per-trial mean), TPR at the calibration 5%-FPR threshold with realised FPR,
    in-sample TPR, counts; flag rates by k. Returns (summary, by_k, {(arm, method, rule, label): draws or AUC})."""
    rows, by_k, store = [], [], {}
    planar = test["N_Planar"].to_numpy()
    for arm, methods in METHODS.items():
        for method in methods:
            for rule in RULES:
                column = score_column(arm, method, rule)
                raw_test, raw_cal = test[column].to_numpy(float), calibration[column].to_numpy(float)
                scores, scores_cal = map_nonfinite(raw_test, rule), map_nonfinite(raw_cal, rule)
                for label, label_column in (("k>=6", "Degenerate"), ("k>=7", "Degenerate_k7"), ("k5_vs_k6", "Degenerate")):
                    mask = np.isin(planar, (5, 6)) if label == "k5_vs_k6" else np.ones(len(test), bool)
                    y, s = test[label_column].to_numpy()[mask], scores[mask]
                    auc, draws = float(weighted_auc(s, y, np.ones((1, mask.sum())))[0]), weighted_auc(s, y, weights[:, mask])
                    store[(arm, method, rule, label)], store[("auc", arm, method, rule, label)] = draws, auc
                    trial_aucs = [roc_auc_score(y[test["Trial"].to_numpy()[mask] == t], s[test["Trial"].to_numpy()[mask] == t])
                                  for t in np.unique(test["Trial"])]
                    row = {"Arm": arm, "Method": method, "Rule": rule, "Label": label, "AUC": auc,
                           "AUC_CI95_Lo": percentile_interval(draws, 0.95)[0], "AUC_CI95_Hi": percentile_interval(draws, 0.95)[1],
                           "Mean_Per_Trial_AUC": float(np.mean(trial_aucs)), "N_Test": int(mask.sum())}
                    if label != "k5_vs_k6":
                        y_cal = calibration[label_column].to_numpy()
                        threshold = calibration_threshold(scores_cal[y_cal == 0])
                        tpr_in, threshold_in = rsb.tpr_at_fpr(y, s, TARGET_FPR)
                        row.update({"AUC_Calibration": roc_auc_score(y_cal, scores_cal), "Threshold_Calibration": threshold,
                                    "TPR_Test_at_Cal_Threshold": float(np.mean(s[y == 1] > threshold)),
                                    "FPR_Test_at_Cal_Threshold": float(np.mean(s[y == 0] > threshold)),
                                    "TPR_InSample_5pct": tpr_in, "FPR_InSample_5pct": float(np.mean(s[y == 0] > threshold_in))})
                        store[("threshold", arm, method, rule, label)] = threshold
                    if label == "k>=6":
                        row.update(mapping_counts(raw_test, y, "test"))
                        row.update(mapping_counts(raw_cal, calibration["Degenerate"].to_numpy(), "calibration"))
                        row["Subset_Counts"] = "; ".join(
                            f"{split} {name}={int(frame[f'{arm.lower()}_{name}'].sum())}" for split, frame in
                            (("test", test), ("calibration", calibration)) for name in SUBSET_COUNT_COLUMNS.get(method, ()))
                    rows.append(row)
                flags = scores > store[("threshold", arm, method, rule, "k>=6")]
                store[("flags", arm, method, rule)] = flags
                for k in range(N_SAMPLE_POINTS + 1):
                    at_k = planar == k
                    rate_draws = weights[:, at_k] @ flags[at_k] / weights[:, at_k].sum(axis=1)
                    by_k.append({"Arm": arm, "Method": method, "Rule": rule, "N_Planar": k, "N_Test": int(at_k.sum()),
                                 "Flag_Rate_at_Cal_Threshold": float(flags[at_k].mean()),
                                 "Flag_Rate_CI95_Lo": percentile_interval(rate_draws, 0.95)[0],
                                 "Flag_Rate_CI95_Hi": percentile_interval(rate_draws, 0.95)[1],
                                 "Median_Mapped_Score": float(np.median(scores[at_k]))})
    return pd.DataFrame(rows), pd.DataFrame(by_k), store


COMPARISONS = ([("H1", ("M8", "capri_coplanarity"), ("M8", "degensac_h_inliers"), 0.95)]
               + [("H2", ("M8", "capri_coplanarity"), ("M8", m), 0.9875)
                  for m in ("capri_kappa", "capri_sigma", "gric_h_minus_f", "svd_condition")]
               + [("H3", ("M8", "capri_coplanarity"), ("M8", m), 0.90) for m in ("degensac_e6", "homography_ls6")]
               + [("H4", ("M8", m), ("M8", "degensac_h_inliers"), 0.975) for m in ("capri_kappa", "capri_ls")]
               + [("D_sigma", ("SC", "capri_sigma"), ("M8", "capri_sigma"), 0.95)]
               + [("D_scene_descriptive", ("SC", m), ("M8", m), 0.95) for m in ("capri_coplanarity", "capri_kappa")])


def comparison_verdict(hypothesis: str, difference: float, lo: float, hi: float, lo95: float, hi95: float) -> str:
    """The pre-written wording of prereg Sections 6.2-6.3 for one comparison."""
    direction = "lower" if difference < 0 else "higher"
    if hypothesis == "H3":
        inside, excludes = -0.01 <= lo and hi <= 0.01, lo95 > 0 or hi95 < 0
        if inside and not excludes:
            return f"equivalent within the +/-0.01 margin (difference {difference:.4f}, 90% CI [{lo:.4f}, {hi:.4f}])"
        if inside:
            return f"{direction} by {abs(difference):.4f} (95% CI excludes 0), within the +/-0.01 equivalence margin"
        if excludes:
            return f"{direction} by {abs(difference):.4f} (95% CI [{lo95:.4f}, {hi95:.4f}])"
        return "inconclusive: neither equivalence nor a difference shown"
    if hypothesis == "D_sigma":
        return ("on these samples capri_sigma scores higher with the scene-level reconstruction (its F and its chart "
                "together; this design does not separate them) than with the 8-point reconstruction, so the synthetic-baseline "
                "comparison gave CAPRI scene-level information the baselines lacked") if lo > 0 else \
            "no dependence on the scene-level reconstruction shown"
    better, worse = {"H1": ("supported", "DEGENSAC better"), "H2": ("capri_coplanarity better", "comparator better"),
                     "H4": ("conditioning score beats DEGENSAC", "DEGENSAC better")}.get(hypothesis, ("first higher", "first lower"))
    return better if lo > 0 else worse if hi < 0 else "no difference shown"


def comparison_table(store: dict, weights: np.ndarray, planar: np.ndarray) -> pd.DataFrame:
    """Paired AUC differences (same bootstrap resamples) for H1-H4, D_sigma and the N5(a) flag-rate criterion."""
    rows = []
    for rule in RULES:
        for hypothesis, (arm_a, method_a), (arm_b, method_b), level in COMPARISONS:
            key_a, key_b = (arm_a, method_a, rule, "k>=6"), (arm_b, method_b, rule, "k>=6")
            auc_a, auc_b, draws = store[("auc",) + key_a], store[("auc",) + key_b], store[key_a] - store[key_b]
            (lo, hi), (lo95, hi95) = percentile_interval(draws, level), percentile_interval(draws, 0.95)
            rows.append({"Hypothesis": hypothesis, "Rule": rule, "Arm_A": arm_a, "Method_A": method_a, "Arm_B": arm_b,
                         "Method_B": method_b, "AUC_A": auc_a, "AUC_B": auc_b, "Difference": auc_a - auc_b, "CI_Level": level,
                         "CI_Lo": lo, "CI_Hi": hi, "CI95_Lo": lo95, "CI95_Hi": hi95,
                         "Verdict": comparison_verdict(hypothesis, auc_a - auc_b, lo, hi, lo95, hi95)})
        for method in ("capri_kappa", "capri_ls"):
            flags = store[("flags", "M8", method, rule)].astype(float)
            high, low = np.isin(planar, (4, 5)), np.isin(planar, (0, 1, 2))
            rate = lambda w, m: (w[:, m] @ flags[m]) / w[:, m].sum(axis=1)
            difference = float(flags[high].mean() - flags[low].mean())
            lo, hi = percentile_interval(rate(weights, high) - rate(weights, low), 0.95)
            rows.append({"Hypothesis": "N5a_flag_rate_k45_minus_k012", "Rule": rule, "Arm_A": "M8", "Method_A": method,
                         "Difference": difference, "CI_Level": 0.95, "CI_Lo": lo, "CI_Hi": hi, "CI95_Lo": lo, "CI95_Hi": hi,
                         "Verdict": "N5(a) confirmed with matched inputs" if lo > 0 else "N5(a) not confirmed"})
    table = pd.DataFrame(rows)
    table["Robust_To_Nonfinite_Rule"] = table.fillna({"Method_B": "", "Arm_B": ""}).groupby(
        ["Hypothesis", "Method_A", "Arm_A", "Method_B", "Arm_B"])["Verdict"].transform(
        lambda v: v.str.replace(r"[-0-9.]+", "#", regex=True).nunique() == 1).to_numpy()
    return table


def stratum_table(test: pd.DataFrame) -> pd.DataFrame:
    """m_hat confusion against k, chi2_3(0.99) exceedance of the true coplanar 6-subset at k = 6,
    and the count of samples where steps 1e-3 and 1e-4 px disagree (relative difference > 1e-3)."""
    rows = []
    for arm in ("m8", "sc"):
        for k in range(N_SAMPLE_POINTS + 1):
            at_k = test[test["N_Planar"] == k]
            for m_hat in range(4, N_SAMPLE_POINTS + 1):
                rows.append({"Arm": arm.upper(), "Table": "m_hat_confusion", "N_Planar": k, "M_hat": m_hat,
                             "Count": int(np.sum(at_k[f"{arm}_m_hat"] == m_hat)), "N": len(at_k)})
            coarse, fine = at_k[f"{arm}_min_T6"].to_numpy(float), at_k[f"{arm}_min_T6_step_1e4"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                disagree = ~(np.abs(coarse / fine - 1.0) <= STEP_AGREEMENT_RTOL) & ~((coarse == fine) | (np.isnan(coarse) & np.isnan(fine)))
            rows.append({"Arm": arm.upper(), "Table": "step_disagreement", "N_Planar": k, "Count": int(disagree.sum()), "N": len(at_k)})
        true_subset = test.loc[test["N_Planar"] == 6, f"{arm}_T_first_six"].to_numpy(float)
        exceed = true_subset > chi2.ppf(0.99, 3)
        rows.append({"Arm": arm.upper(), "Table": "chi2_3_0.99_exceedance_true_subset", "N_Planar": 6,
                     "Count": int(exceed.sum()), "N": len(true_subset), "Value": float(exceed.mean()),
                     "Mean_T": float(np.mean(true_subset)), "Median_T": float(np.median(true_subset))})
    return pd.DataFrame(rows)


def print_method_runtimes(n_samples: int) -> None:
    """Runtime per scorer on generic 8-point samples of calibration scene 0 (stdout only, never saved)."""
    view_1, view_2, _ = rsb.make_scene(make_rng(EXPERIMENT_SEED + SPLITS["calibration"][0]))
    for name, scorer in (("all M8 scores", score_matched_sample), ("degensac_h_inliers", rsb.score_degensac),
                         ("gric_h_minus_f", rsb.score_gric), ("svd_condition", rsb.score_svd),
                         ("70 homography fits (degensac_e6)", lambda a, b: homography_subset_errors(a, b, SUBSETS_4))):
        start = time.perf_counter()
        for first in range(0, 8 * n_samples, 8):
            scorer(view_1[first:first + 8], view_2[first:first + 8])
        print(f"  runtime {name}: {1e3 * (time.perf_counter() - start) / n_samples:.2f} ms per sample")


def run_experiment() -> None:
    """Collect all samples, analyse, and write the five result files."""
    n_workers = int(os.environ.get("CAPRI_N_WORKERS", os.cpu_count() or 1))
    start = time.perf_counter()
    samples = collect_samples(n_workers)
    print(f"  collected {len(samples)} samples in {time.perf_counter() - start:.1f} s with {n_workers} workers")
    test = samples[samples["Split"] == "test"].reset_index(drop=True)
    calibration = samples[samples["Split"] == "calibration"].reset_index(drop=True)
    weights = bootstrap_sample_weights(test)
    summary, by_k, store = score_summary(test, calibration, weights)
    comparisons = comparison_table(store, weights, test["N_Planar"].to_numpy())
    stratum = stratum_table(test)
    print(f"  total runtime {time.perf_counter() - start:.1f} s")
    print_method_runtimes(36)
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(summary.loc[(summary["Rule"] == "primary") & (summary["Label"] == "k>=6"),
                          ["Arm", "Method", "AUC", "AUC_CI95_Lo", "AUC_CI95_Hi", "TPR_Test_at_Cal_Threshold"]].to_string(index=False))
        print(comparisons[["Hypothesis", "Rule", "Method_A", "Method_B", "Difference", "CI_Lo", "CI_Hi", "Verdict"]].to_string(index=False))
    for frame, name, description in (
            (samples, "degeneracy_matched_samples.csv", "matched degeneracy experiment: per-sample degeneracy scores, arms M8 and SC (raw, before non-finite mapping)"),
            (summary, "degeneracy_matched_summary.csv", "matched degeneracy experiment: AUC with paired cluster-bootstrap CI, TPR at calibration 5%-FPR threshold, counts"),
            (comparisons, "degeneracy_matched_comparisons.csv", "matched degeneracy experiment: pre-registered paired comparisons H1-H4, D_sigma, N5(a)"),
            (by_k, "degeneracy_matched_by_planar_count.csv", "matched degeneracy experiment: flag rate at the calibration threshold by number of planar points"),
            (stratum, "degeneracy_matched_stratum.csv", "matched degeneracy experiment: m_hat confusion, chi2 exceedance, step-size check")):
        save_results(frame, name, EXPERIMENT_SEED, description)


if __name__ == "__main__":
    run_experiment()
