"""
forensics/video/multiview.py

Cross-subset invariant discrepancy Delta_J (constraint: a rigid scene has the
same invariants in every reconstruction).

For a rigid static scene, two projective reconstructions of the same tracks
from DISJOINT view subsets differ from the true scene by 4x4 homographies, so
every six-tuple has identical invariants in both: Delta_J = log|J(A)| - log|J(B)|
is zero up to noise. The two estimates are independent by construction, which
is what lets Delta_J escape Corollary 2.6 (rho is zero for any coherent point set).

Reconstruction per subset of three views (Hartley-Zisserman):
  1. normalized image coordinates (centre, divide by max(H, W) / 2);
  2. F between the first two views (USAC-MAGSAC) -> P1 = [I | 0],
     P2 = [[e2]x F | e2];
  3. DLT triangulation of every track;
  4. DLT resection of the third camera;
  5. projective bundle adjustment of P2, P3 and all points (P1 fixed as gauge),
     minimizing reprojection error with a sparse least-squares solver.
All tracks (at least min_tracks_per_window, default 20) are reconstructed;
six points alone are never used (minimal six-point/three-view data can have
up to three reconstructions; Quan 1995, to be verified before citing).
"""

from __future__ import annotations
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from forensics.geometry.invariants import kappa_j, log_abs_invariants, sigma4


def dlt_triangulate(points_2d: list[np.ndarray], cameras: list[np.ndarray]) -> np.ndarray:
    """(4, N) homogeneous points (unit norm) from >= 2 views by linear triangulation."""
    n = points_2d[0].shape[0]
    X = np.zeros((4, n))
    for i in range(n):
        rows = []
        for x, P in zip(points_2d, cameras):
            rows.append(x[i, 0] * P[2] - P[0])
            rows.append(x[i, 1] * P[2] - P[1])
        _, _, vt = np.linalg.svd(np.asarray(rows))
        X[:, i] = vt[-1] / np.linalg.norm(vt[-1])
    return X


def dlt_resection(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """3x4 camera from (4, N) points and (N, 2) projections."""
    rows = []
    for i in range(X.shape[1]):
        Xi = X[:, i]
        rows.append(np.r_[np.zeros(4), -Xi, x[i, 1] * Xi])
        rows.append(np.r_[Xi, np.zeros(4), -x[i, 0] * Xi])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    P = vt[-1].reshape(3, 4)
    return P / np.linalg.norm(P)


def cameras_from_fundamental(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    F = F / np.linalg.norm(F)          # OpenCV may return F with norm ~1e8 on exact data
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    _, _, vt = np.linalg.svd(F.T)
    e2 = vt[-1]
    skew = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    return P1, np.hstack([skew @ F, e2[:, None]])


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = P @ X
    return (x[:2] / x[2]).T


def bundle_adjust(cameras: list[np.ndarray], X: np.ndarray, observations: list[np.ndarray],
                  max_nfev: int = 200) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Projective BA with the first camera fixed. Returns cameras, points (unit columns), RMS reprojection."""
    n_views, n_points = len(cameras), X.shape[1]
    n_cam_params = 12 * (n_views - 1)

    def unpack(params):
        cams = [cameras[0]] + [params[12 * k:12 * (k + 1)].reshape(3, 4) for k in range(n_views - 1)]
        pts = params[n_cam_params:].reshape(4, n_points, order="F")
        return cams, pts

    def residuals(params):
        cams, pts = unpack(params)
        return np.concatenate([(_project(P, pts) - obs).ravel() for P, obs in zip(cams, observations)])

    sparsity = lil_matrix((2 * n_points * n_views, n_cam_params + 4 * n_points), dtype=int)
    for v in range(n_views):
        for i in range(n_points):
            row = 2 * (v * n_points + i)
            if v > 0:
                sparsity[row:row + 2, 12 * (v - 1):12 * v] = 1
            sparsity[row:row + 2, n_cam_params + 4 * i:n_cam_params + 4 * (i + 1)] = 1
    x0 = np.concatenate([P.ravel() for P in cameras[1:]] + [X.ravel(order="F")])
    solution = least_squares(residuals, x0, jac_sparsity=sparsity, method="trf", x_scale="jac", max_nfev=max_nfev)
    cams, pts = unpack(solution.x)
    pts = pts / np.linalg.norm(pts, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(solution.fun ** 2)))
    return cams, pts, rms


def projective_reconstruction(views: list[np.ndarray], seed: int = 0, ransac_threshold: float = 1e-3) -> dict:
    """
    Projective reconstruction of N tracks seen in three views.

    views: list of three (N, 2) arrays in NORMALIZED image coordinates.
    Returns {"X": (4, N) points in the whitened frame, "cameras": [...] (BA frame),
             "rms": reprojection RMS (normalized units)}.
    """
    import cv2

    assert len(views) == 3, "a subset has three views"
    cv2.setRNGSeed(seed)
    try:
        F, _ = cv2.findFundamentalMat(views[0].astype(np.float64), views[1].astype(np.float64),
                                      cv2.USAC_MAGSAC, ransac_threshold, 0.9999)
    except cv2.error:  # USAC can fail an internal assertion on degenerate input: no model
        F = None
    if F is None or F.shape[0] < 3:
        raise RuntimeError("fundamental matrix estimation failed")
    P1, P2 = cameras_from_fundamental(F[:3])
    X = dlt_triangulate(views[:2], [P1, P2])
    P3 = dlt_resection(X, views[2])
    cameras, X, rms = bundle_adjust([P1, P2, P3], X, views)
    return {"X": whiten_projective_frame(X), "cameras": cameras, "rms": rms}


def whiten_projective_frame(X: np.ndarray) -> np.ndarray:
    """
    Canonical projective normalization (Tyler shape matrix; see
    algorithms.utils.whiten_homogeneous_points). J_i are unchanged by any 4x4
    linear map, but the raw frame from P1 = [I | 0] can be arbitrarily skewed,
    which makes sigma_4 and kappa_J meaningless. After this normalization the
    two subset reconstructions are in frames that agree up to an orthogonal map
    (exactly, for noise-free data), so the gates mean the same thing in both.
    """
    from algorithms.utils import whiten_homogeneous_points

    return whiten_homogeneous_points(X)


def sample_gated_tuples(X_a: np.ndarray, X_b: np.ndarray, n_tuples: int, rng: np.random.Generator,
                        sigma4_min: float, max_attempts_factor: int = 20) -> list[tuple[int, ...]]:
    """Six-tuples whose every 4-subset has sigma_4 >= sigma4_min in BOTH reconstructions."""
    n = X_a.shape[1]
    chosen, attempts = [], 0
    while len(chosen) < n_tuples and attempts < max_attempts_factor * n_tuples:
        attempts += 1
        t = tuple(sorted(rng.choice(n, size=6, replace=False).tolist()))
        if all(min(sigma4(X[:, list(q)]) for q in combinations(t, 4)) >= sigma4_min for X in (X_a, X_b)):
            chosen.append(t)
    return chosen


def delta_j(X_a: np.ndarray, X_b: np.ndarray, tuples: list[tuple[int, ...]]) -> dict:
    """Delta_J per tuple, the kappa-based scale, and the standardized magnitude."""
    deltas, scales = [], []
    for t in tuples:
        Pa, Pb = X_a[:, list(t)], X_b[:, list(t)]
        deltas.append(log_abs_invariants(Pa) - log_abs_invariants(Pb))
        scales.append(kappa_j(Pa) + kappa_j(Pb))
    deltas = np.asarray(deltas)
    scales = np.asarray(scales)
    magnitude = np.max(np.abs(deltas), axis=1) if len(deltas) else np.zeros(0)
    return {"delta": deltas, "scale": scales,
            "standardized": magnitude / np.where(scales > 0, scales, np.inf)}
