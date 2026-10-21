"""
experiments/run_synthetic_baselines.py

Synthetic baseline comparison. Numpy + OpenCV only;
no PyTorch.

Design choices
--------------
  1. Scene geometry: all points lie 4-12 units in front of the first camera,
     and the image noise is in pixels of calibrated cameras (points in a cube
     centred on the first camera would put many behind it or at near-zero
     depth).
  2. Ground truth per 8-point sample, not per point: a rule that counts a
     sample as degenerate only when all 8 points are degenerate has
     probability 2.3e-6 per sample with 20% degenerate points, so its leak
     rate would almost always be 0/0.
  3. CAPRI scores come from the triangulated points of the sample; no learned
     fusion model and no fixed dummy inputs are used.
  4. All methods are compared on the same unit (samples) and over all
     thresholds (AUC), plus a fixed operating point each.

Design
------
Scene: 80% generic points and 20% points within 1e-3 (relative) of one plane,
all 4-12 units in front of camera 1; two calibrated cameras (f = 800 px,
1280 x 960 px, second camera rotated 0.3 rad and translated); Gaussian
image noise sigma = 0.5 px.

Unit of decision: an 8-point minimal sample (what an 8-point RANSAC draws).
Ground truth: a sample is DEGENERATE if at least 6 of its 8 points lie on the
dominant plane (a homography explains them, so the fundamental matrix is
ill-determined). Samples are drawn with k = 0..8 planar points, 50 per k per
trial, so both classes are well represented (class balance is reported).

CAPRI input: all correspondences of a trial -> USAC-MAGSAC
fundamental matrix in Hartley-normalized image coordinates -> projective
cameras P1 = [I | 0], P2 = [[e2]x F | e2] -> DLT triangulation. No calibration
and no ground-truth 3D points are used by CAPRI.

Scores (higher = more degenerate), each computed on the same samples:
  capri_ls            max L(S) over the 28 six-point subsets of the sample
  capri_kappa         max kappa_J over the same 28 subsets (docs/MATHEMATICS.md,
                      Section 5; the pipeline's default conditioning metric)
  capri_sigma         1 - min over subsets of sigma_4/sigma_1 (scale-free)
  svd_condition       sigma_1 / sigma_8 of the Hartley-normalized 8x9 design matrix
  degensac_h_inliers  max number of the 8 points within 2 px transfer error of a
                      homography fitted to any 4 of them (DEGENSAC-style test)
  gric_h_minus_f      GRIC(F) - GRIC(H) (Torr's model selection; positive = H preferred)

Operating points also reported: CAPRI's fixed threshold L(S) < 1000 (accept;
data label "manuscript" in the column and row names) and each method's
threshold at 5% false-positive rate.

Per-sample scores are written to true_synthetic_baselines.csv; the summary
(AUC with bootstrap CI over trials, TPR at 5% FPR) to
true_synthetic_baselines_summary.csv. Seed 42100 + trial.
"""

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import get_experiment_seed, make_rng, save_results
from algorithms.conditioning.scores import (
    batch_invariant_noise_gain,
    compute_conditioning_score,
    compute_singular_value_conditioning,
)
from reconstruction.triangulation import (
    estimate_projection_matrices_from_fundamental,
    triangulate_point_cloud,
)

EXPERIMENT_SEED = get_experiment_seed(100)  # 42100
N_TRIALS = int(__import__("os").environ.get("CAPRI_N_TRIALS", 30))
N_POINTS = 1000
PLANAR_FRACTION = 0.2
NOISE_PX = 0.5
SAMPLES_PER_K = 50
DEGENERATE_MIN_PLANAR = 6
H_INLIER_PX = 2.0
FOCAL_PX, WIDTH_PX, HEIGHT_PX = 800.0, 1280, 960
SCORES = ["capri_ls", "capri_kappa", "capri_sigma", "svd_condition", "degensac_h_inliers", "gric_h_minus_f"]
SUBSETS_6_OF_8 = list(combinations(range(8), 6))
SUBSETS_4_OF_8 = list(combinations(range(8), 4))


# --------------------------------------------------------------------------- scene

def make_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (image points view 1 (N,2), view 2 (N,2), planar mask (N,))."""
    n_planar = int(N_POINTS * PLANAR_FRACTION)
    generic = np.column_stack([rng.uniform(-3, 3, N_POINTS - n_planar),
                               rng.uniform(-2, 2, N_POINTS - n_planar),
                               rng.uniform(4, 12, N_POINTS - n_planar)])
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    centre = np.array([0.0, 0.0, 8.0])
    basis_u = np.cross(normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0])
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    coefficients = rng.uniform(-3, 3, size=(n_planar, 2))
    planar = centre + coefficients[:, :1] * basis_u + coefficients[:, 1:] * basis_v
    planar += np.outer(rng.uniform(-1e-3, 1e-3, n_planar) * 8.0, normal)
    world = np.vstack([generic, planar])
    planar_mask = np.r_[np.zeros(N_POINTS - n_planar, bool), np.ones(n_planar, bool)]

    K = np.array([[FOCAL_PX, 0, WIDTH_PX / 2], [0, FOCAL_PX, HEIGHT_PX / 2], [0, 0, 1.0]])
    angle = 0.3
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    t = np.array([-2.0, 0.2, 0.5])

    def project(points_cam: np.ndarray) -> np.ndarray:
        projected = (K @ points_cam.T).T
        return projected[:, :2] / projected[:, 2:3]

    view_1 = project(world) + rng.normal(0, NOISE_PX, (N_POINTS, 2))
    view_2 = project(world @ R.T + t) + rng.normal(0, NOISE_PX, (N_POINTS, 2))
    return view_1, view_2, planar_mask


def hartley_normalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(points - centroid, axis=1))
    T = np.array([[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1.0]])
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ T.T
    return homogeneous[:, :2], T


def projective_reconstruction(view_1: np.ndarray, view_2: np.ndarray, seed: int) -> np.ndarray:
    """F (USAC-MAGSAC, normalized coordinates) -> P1, P2 -> DLT points (4, N)."""
    norm_1, _ = hartley_normalize(view_1)
    norm_2, _ = hartley_normalize(view_2)
    cv2.setRNGSeed(seed)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(view_1 - view_1.mean(axis=0), axis=1))
    F, _ = cv2.findFundamentalMat(norm_1, norm_2, cv2.USAC_MAGSAC, 1.0 * scale, 0.999)
    P1, P2 = estimate_projection_matrices_from_fundamental(F[:3], norm_1, norm_2)
    return triangulate_point_cloud([norm_1, norm_2], np.stack([P1, P2]))


# --------------------------------------------------------------------------- scores

def score_capri(points_3d: np.ndarray, sample: np.ndarray) -> tuple[float, float]:
    ls_values, sigma_values = [], []
    for subset in SUBSETS_6_OF_8:
        t = tuple(int(sample[i]) for i in subset)
        ls_values.append(compute_conditioning_score(points_3d, t))
        sigma_values.append(compute_singular_value_conditioning(points_3d, t))
    ls_values = np.asarray(ls_values, dtype=float)
    max_ls = float(np.max(np.where(np.isfinite(ls_values), ls_values, 1e300)))
    return max_ls, 1.0 - float(np.min(sigma_values))


def score_capri_kappa(points_3d: np.ndarray, sample: np.ndarray) -> float:
    tuples = np.array([[int(sample[i]) for i in subset] for subset in SUBSETS_6_OF_8], dtype=np.int64)
    kappa = batch_invariant_noise_gain(points_3d, tuples)
    return float(np.max(np.where(np.isfinite(kappa), kappa, 1e300)))


def score_svd(view_1: np.ndarray, view_2: np.ndarray) -> float:
    n1, _ = hartley_normalize(view_1)
    n2, _ = hartley_normalize(view_2)
    design = np.column_stack([n2[:, 0] * n1[:, 0], n2[:, 0] * n1[:, 1], n2[:, 0],
                              n2[:, 1] * n1[:, 0], n2[:, 1] * n1[:, 1], n2[:, 1],
                              n1[:, 0], n1[:, 1], np.ones(len(n1))])
    singular = np.linalg.svd(design, compute_uv=False)
    return float(singular[0] / max(singular[7], 1e-300))


def fit_homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    return vt[-1].reshape(3, 3)


def transfer_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    mapped = np.column_stack([src, np.ones(len(src))]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        mapped = mapped[:, :2] / mapped[:, 2:3]
    errors = np.linalg.norm(mapped - dst, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def score_degensac(view_1: np.ndarray, view_2: np.ndarray) -> float:
    best = 0
    for subset in SUBSETS_4_OF_8:
        idx = list(subset)
        H = fit_homography_dlt(view_1[idx], view_2[idx])
        best = max(best, int(np.sum(transfer_errors(H, view_1, view_2) < H_INLIER_PX)))
    return float(best)


def score_gric(view_1: np.ndarray, view_2: np.ndarray) -> float:
    """GRIC(F) - GRIC(H) with Torr's constants (d=3 for F, d=2 for H; r=4 data dims)."""
    n, sigma2, r = len(view_1), NOISE_PX ** 2, 4
    lambda_1, lambda_2, lambda_3 = np.log(r), np.log(r * n), 2.0

    def gric(residuals_sq: np.ndarray, d: int, k: int) -> float:
        rho = np.minimum(residuals_sq / sigma2, lambda_3 * (r - d))
        return float(np.sum(rho) + lambda_1 * d * n + lambda_2 * k)

    H = fit_homography_dlt(view_1, view_2)
    h_residuals = transfer_errors(H, view_1, view_2) ** 2
    F, _ = cv2.findFundamentalMat(view_1, view_2, cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        return float("inf")
    x1 = np.column_stack([view_1, np.ones(n)])
    x2 = np.column_stack([view_2, np.ones(n)])
    Fx1, Ftx2 = x1 @ F.T, x2 @ F
    numerator = np.sum(x2 * Fx1, axis=1) ** 2
    denominator = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    f_residuals = numerator / np.maximum(denominator, 1e-300)
    return gric(f_residuals, d=3, k=7) - gric(h_residuals, d=2, k=8)


# --------------------------------------------------------------------------- experiment

def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> tuple[float, float]:
    negatives = np.sort(scores[labels == 0])
    threshold = negatives[int(np.ceil((1 - fpr) * len(negatives))) - 1]
    return float(np.mean(scores[labels == 1] > threshold)), float(threshold)


def run_experiment() -> pd.DataFrame:
    print("=" * 60)
    print("SYNTHETIC BASELINES: DEGENERATE 8-POINT SAMPLE DETECTION")
    print("=" * 60)
    rows = []
    for trial in range(N_TRIALS):
        seed = EXPERIMENT_SEED + trial
        rng = make_rng(seed)
        view_1, view_2, planar = make_scene(rng)
        points_3d = projective_reconstruction(view_1, view_2, seed)
        planar_idx, generic_idx = np.flatnonzero(planar), np.flatnonzero(~planar)
        for k in range(9):
            for _ in range(SAMPLES_PER_K):
                sample = np.r_[rng.choice(planar_idx, k, replace=False),
                               rng.choice(generic_idx, 8 - k, replace=False)]
                capri_ls, capri_sigma = score_capri(points_3d, sample)
                rows.append({
                    "Trial": trial, "Seed": seed, "N_Planar_In_Sample": k,
                    "Degenerate": int(k >= DEGENERATE_MIN_PLANAR),
                    "capri_ls": capri_ls, "capri_kappa": score_capri_kappa(points_3d, sample),
                    "capri_sigma": capri_sigma,
                    "svd_condition": score_svd(view_1[sample], view_2[sample]),
                    "degensac_h_inliers": score_degensac(view_1[sample], view_2[sample]),
                    "gric_h_minus_f": score_gric(view_1[sample], view_2[sample]),
                    "capri_accept_manuscript": bool(capri_ls < 1000.0),
                })
        print(f"  trial {trial + 1}/{N_TRIALS} done")
    df = pd.DataFrame(rows)

    summary = []
    labels = df["Degenerate"].to_numpy()
    for score in SCORES:
        values = df[score].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, np.nanmax(values[np.isfinite(values)]))
        per_trial_auc = [roc_auc_score(g["Degenerate"], np.nan_to_num(g[score].to_numpy(float), posinf=1e300))
                         for _, g in df.groupby("Trial")]
        boot = make_rng(EXPERIMENT_SEED + 999)
        boot_means = [np.mean(boot.choice(per_trial_auc, len(per_trial_auc))) for _ in range(2000)]
        tpr5, thr5 = tpr_at_fpr(labels, values, 0.05)
        summary.append({
            "Method": score, "AUC": roc_auc_score(labels, values),
            "AUC_CI_Lo": float(np.percentile(boot_means, 2.5)),
            "AUC_CI_Hi": float(np.percentile(boot_means, 97.5)),
            "TPR_at_5pct_FPR": tpr5, "Threshold_at_5pct_FPR": thr5,
            "N_Samples": len(df), "Degenerate_Fraction": float(labels.mean()),
        })
    manuscript_flag = df["capri_accept_manuscript"].to_numpy()
    summary.append({
        "Method": "capri_ls at manuscript threshold (accept if L<1000)",
        "AUC": np.nan, "AUC_CI_Lo": np.nan, "AUC_CI_Hi": np.nan,
        "TPR_at_5pct_FPR": np.nan, "Threshold_at_5pct_FPR": 1000.0,
        "N_Samples": len(df), "Degenerate_Fraction": float(labels.mean()),
        "Accept_Rate_NonDegenerate": float(manuscript_flag[labels == 0].mean()),
        "Accept_Rate_Degenerate": float(manuscript_flag[labels == 1].mean()),
    })
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))

    save_results(df, "true_synthetic_baselines.csv", EXPERIMENT_SEED,
                 "Synthetic baselines: per-sample degeneracy scores")
    save_results(summary, "true_synthetic_baselines_summary.csv", EXPERIMENT_SEED,
                 "Synthetic baselines: AUC (bootstrap CI over trials) and TPR at 5% FPR")
    return df


if __name__ == "__main__":
    run_experiment()
