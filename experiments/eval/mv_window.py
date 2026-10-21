"""
experiments/eval/mv_window.py

Every statistic of docs/prereg/evaluation_plan.md (the evaluation plan) for one window of six views,
computed once so that all statistics see the same reconstructions.

Subsets: A = views {0, 2, 4}, B = views {1, 3, 5} (interleaved; detector benchmark also uses the
contiguous split {0, 1, 2} / {3, 4, 5}). Each subset is reconstructed as in
forensics/video/multiview.py (USAC-MAGSAC F, canonical cameras, DLT, resection,
projective bundle adjustment by LM, forensics/video/fast_ba.py); the six-view joint bundle adjustment starts from
subset A and resected cameras of B. Coordinates are normalized pixels
((x - centre) / 640); sums of squares are converted back to pixels.

Statistics (larger = more inconsistent):
  pencil_q10 / q50 / q90   quantiles of |d log|CR|| over 50,000 sampled pencils
  asinh_q10                Q_0.1 of |d asinh CR| over 20,000 sampled pencils
  acat                     -log10 ACAT p of standardized pencil discrepancies
  gauge_q90 / gauge_med    additive gauge fit on 20,000 sampled log-brackets
  deltaj_med               legacy: median standardized max|Delta_J| (100 gated tuples)
  registration_rms         RMS sine residual of a DLT projective registration
  split_glrt               (G_all - G_A - G_B) / sigma^2, sigma = generating noise
  f_ratio, f_ratio_p       sigma-free F-ratio, df1 = 3n - 15, df2 = 2 (3n - 18)
  glrt_all                 G_all / sigma^2 (dof 9n - 51)
  sampson_mean             mean squared Sampson error (px^2), MAGSAC F per consecutive pair
  magsac_outlier           1 - mean MAGSAC inlier ratio (1 px), consecutive pairs
Abstention: reconstruction failure, or a view-0/1 homography (MAGSAC, 2 px) with
>= 90% inliers (planar scene).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from algorithms.consistency.pencil import pencil_discrepancies  # noqa: E402
from algorithms.consistency import variants  # noqa: E402
from algorithms.degeneracy.coplanarity import subsets_of_size  # noqa: E402
from algorithms.utils import whiten_homogeneous_points  # noqa: E402
from forensics.stats.combine import cauchy_combination  # noqa: E402
from forensics.video.fast_ba import bundle_adjust_lm  # noqa: E402
from forensics.video.multiview import (cameras_from_fundamental, delta_j, dlt_resection,  # noqa: E402
                                       dlt_triangulate, sample_gated_tuples)
from synthetic_data.injection import HEIGHT, WIDTH  # noqa: E402
import lab  # noqa: E402

HALF = max(WIDTH, HEIGHT) / 2.0
CENTRE = np.array([WIDTH / 2.0, HEIGHT / 2.0])
SPLITS = {"interleaved": ((0, 2, 4), (1, 3, 5)), "contiguous": ((0, 1, 2), (3, 4, 5))}
N_PENCILS, N_PENCILS_SMALL, N_GAUGE_QUADS = 50_000, 20_000, 20_000
ACAT_NOISE_SCALE = 2.198e-4     # m0: median(D/g)/0.6745 over 40 rigid windows at 0.5 px, exploration seeds (evaluation_plan)
SIGMA4_MIN = 0.02               # configs/default.yaml video.sigma4_min (legacy Delta_J gate)
PLANAR_INLIER_FRACTION = 0.9


SMOKE = bool(os.environ.get("EVAL_SMOKE"))   # smoke test: tiny grids, exploration seeds (+60,000,000), smoke_* files
SMOKE_SEED_OFFSET = 60_000_000


def out_name(name: str) -> str:
    return f"smoke_{name}" if SMOKE else name


def shrink(count: int, smoke_count: int = 6) -> int:
    return min(count, smoke_count) if SMOKE else count


def normalize(tracks_px: np.ndarray) -> np.ndarray:
    return (tracks_px - CENTRE) / HALF


def reconstruct(views: list[np.ndarray], seed: int) -> dict:
    """Projective reconstruction of three normalized views; X in the BA frame (unit columns)."""
    cv2.setRNGSeed(seed)
    F, _ = cv2.findFundamentalMat(views[0], views[1], cv2.USAC_MAGSAC, 1e-3, 0.9999)
    if F is None or F.shape[0] < 3:
        raise RuntimeError("fundamental matrix estimation failed")
    P1, P2 = cameras_from_fundamental(F[:3])
    X = dlt_triangulate(views[:2], [P1, P2])
    cameras, X, sse = bundle_adjust_lm([P1, P2, dlt_resection(X, views[2])], X, views)
    return {"X": X, "cameras": cameras, "sse_px": sse * HALF ** 2}


def joint_sse(rec_a: dict, views_all: list[np.ndarray], order: tuple[int, ...]) -> float:
    """Six-view projective BA started from subset A's cameras and points; views in `order`."""
    obs = [views_all[v] for v in order]
    cameras = list(rec_a["cameras"]) + [dlt_resection(rec_a["X"], obs[k]) for k in range(3, 6)]
    _, _, sse = bundle_adjust_lm(cameras, rec_a["X"], obs)
    return sse * HALF ** 2


def pair_baselines(tracks_px: np.ndarray, seed: int, moved: np.ndarray | None = None) -> tuple[float, float, float]:
    """Mean squared Sampson error (px^2), 1 - mean inlier ratio of MAGSAC F over consecutive pairs, and the
    fraction of moved tracks that MAGSAC marks as an outlier in at least one pair (NaN without `moved`)."""
    sampson, inliers, outlier_any = [], [], np.zeros(tracks_px.shape[1], dtype=bool)
    for v in range(tracks_px.shape[0] - 1):
        x1, x2 = tracks_px[v].astype(np.float64), tracks_px[v + 1].astype(np.float64)
        cv2.setRNGSeed(seed + v)
        F, mask = cv2.findFundamentalMat(x1, x2, cv2.USAC_MAGSAC, 1.0, 0.9999)
        if F is None or F.shape[0] < 3:
            return float("nan"), float("nan"), float("nan")
        outlier_any |= ~mask.ravel().astype(bool)
        h1, h2 = np.c_[x1, np.ones(len(x1))], np.c_[x2, np.ones(len(x2))]
        Fx1, Ftx2 = h1 @ F[:3].T, h2 @ F[:3]
        num = np.sum(h2 * Fx1, axis=1) ** 2
        sampson.append(num / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2))
        inliers.append(float(mask.mean()))
    removed = float(outlier_any[moved].mean()) if moved is not None and moved.any() else float("nan")
    return float(np.mean(np.concatenate(sampson))), float(1.0 - np.mean(inliers)), removed


def is_planar(tracks_px: np.ndarray, seed: int) -> bool:
    cv2.setRNGSeed(seed)
    H, mask = cv2.findHomography(tracks_px[0], tracks_px[1], cv2.USAC_MAGSAC, 2.0, maxIters=5000, confidence=0.999)
    return H is not None and float(mask.mean()) >= PLANAR_INLIER_FRACTION


GUARDED = (np.linalg.LinAlgError, ValueError, FloatingPointError, cv2.error)


def _guarded(row: dict, errors: list, name: str, keys: tuple[str, ...], compute) -> None:
    """Run one statistic group; on a numerical failure its keys are NaN and the error is logged."""
    try:
        row.update(compute())
    except GUARDED as error:
        row.update({key: float("nan") for key in keys})
        errors.append(f"{name}: {type(error).__name__}")


def pair_statistics(X_a: np.ndarray, X_b: np.ndarray, seed: int, n_pencils: int = N_PENCILS) -> dict:
    """Frame-free statistics between two reconstructions of the same n points (any frames)."""
    n = X_a.shape[1]
    row, errors = {}, []

    def pencil():
        D = pencil_discrepancies(X_a, X_b, max_pencils=n_pencils, seed=seed)
        D = D[np.isfinite(D)]
        return {"pencil_q10": float(np.quantile(D, 0.1)), "pencil_q50": float(np.quantile(D, 0.5)),
                "pencil_q90": float(np.quantile(D, 0.9))}

    def asinh_and_acat():
        small = variants.sampled_pencils(n, N_PENCILS_SMALL, seed + 1)
        asinh = variants.asinh_discrepancies(X_a, X_b, small)
        z = variants.standardized_discrepancies(X_a, X_b, small, ACAT_NOISE_SCALE)
        z = z[np.isfinite(z)]
        p_acat = cauchy_combination(2.0 * stats.norm.sf(z))
        return {"asinh_q10": float(np.nanquantile(asinh, 0.1)), "acat": float(-np.log10(max(p_acat, 1e-300))),
                "acat_scale": float(np.median(z)) if z.size else float("nan")}

    def gauge():
        all_quads = subsets_of_size(n, 4)
        pick = np.random.default_rng(seed + 2).choice(len(all_quads), min(N_GAUGE_QUADS, len(all_quads)), replace=False)
        fit = variants.gauge_fit(X_a, X_b, all_quads[np.sort(pick)])
        return {"gauge_q90": fit["statistic"], "gauge_med": fit["median"]}

    def registration():
        _, r = lab.fit_projective(whiten_homogeneous_points(X_a), whiten_homogeneous_points(X_b))
        return {"registration_rms": float(np.sqrt(np.mean(r ** 2)))}

    _guarded(row, errors, "pencil", ("pencil_q10", "pencil_q50", "pencil_q90"), pencil)
    _guarded(row, errors, "asinh_acat", ("asinh_q10", "acat", "acat_scale"), asinh_and_acat)
    _guarded(row, errors, "gauge", ("gauge_q90", "gauge_med"), gauge)
    _guarded(row, errors, "registration", ("registration_rms",), registration)
    row["errors"] = "; ".join(errors)
    return row


def window_statistics(tracks_px: np.ndarray, noise_px: float, seed: int, split: str = "interleaved",
                      with_legacy: bool = True, moved: np.ndarray | None = None) -> dict:
    """All statistics of one six-view window; {"abstain": True, ...} when the window abstains."""
    cv2.setNumThreads(1)
    n = tracks_px.shape[1]
    row = {"abstain": False, "abstain_reason": ""}
    try:
        planar = is_planar(tracks_px, seed)
    except cv2.error as error:
        return {**row, "abstain": True, "abstain_reason": f"homography: {str(error).splitlines()[-1][:80]}"}
    if planar:
        return {**row, "abstain": True, "abstain_reason": "planar"}
    views = [normalize(t) for t in tracks_px]
    a_idx, b_idx = SPLITS[split]
    try:
        rec_a = reconstruct([views[v] for v in a_idx], seed)
        rec_b = reconstruct([views[v] for v in b_idx], seed)
        g_all = joint_sse(rec_a, views, a_idx + b_idx)
    except (RuntimeError, np.linalg.LinAlgError, ValueError, cv2.error) as error:
        return {**row, "abstain": True, "abstain_reason": f"reconstruction: {error}"}
    row.update(pair_statistics(rec_a["X"], rec_b["X"], seed))
    df_sub, df_all = 3 * n - 18, 9 * n - 51
    df1, df2 = df_all - 2 * df_sub, 2 * df_sub
    g_split = g_all - rec_a["sse_px"] - rec_b["sse_px"]
    f_ratio = (g_split / df1) / ((rec_a["sse_px"] + rec_b["sse_px"]) / df2)
    sigma2 = max(noise_px, 1e-12) ** 2
    row.update({"split_glrt": g_split / sigma2, "glrt_all": g_all / sigma2, "f_ratio": f_ratio,
                "f_ratio_p": float(stats.f.sf(f_ratio, df1, df2)),
                "sigma_hat_px": float(np.sqrt((rec_a["sse_px"] + rec_b["sse_px"]) / df2)),
                "nested_violation": bool(g_split < -1e-6 * g_all)})
    errors = [row.pop("errors")] if row.get("errors") else []
    pair_keys = ("sampson_mean", "magsac_outlier", "magsac_removed_moved")
    _guarded(row, errors, "pairs", pair_keys, lambda: dict(zip(pair_keys, pair_baselines(tracks_px, seed, moved))))

    def legacy():
        rng = np.random.default_rng(seed)
        Xa, Xb = whiten_homogeneous_points(rec_a["X"]), whiten_homogeneous_points(rec_b["X"])
        tuples = sample_gated_tuples(Xa, Xb, 100, rng, SIGMA4_MIN)
        return {"deltaj_med": float(np.median(delta_j(Xa, Xb, tuples)["standardized"])) if tuples else float("nan")}

    if with_legacy:
        _guarded(row, errors, "deltaj", ("deltaj_med",), legacy)
    row["errors"] = "; ".join(errors)
    row["_rec"] = (rec_a, rec_b)
    return row
