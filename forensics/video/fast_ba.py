"""
forensics/video/fast_ba.py

Projective bundle adjustment by Levenberg-Marquardt with an analytic dense
Jacobian (Hartley-Zisserman 2nd ed., Appendix A6; cited, nothing new). Used by
the evaluation harness (docs/prereg/evaluation_plan.md), where scipy's
finite-difference trust-region solver in multiview.bundle_adjust was too slow
(2-15 s per six-view window) and stopped early on inconsistent windows.

Model: x_vi = dehom(P_v X_i), residual r_vi = dehom(P_v X_i) - x_vi (normalized
image units). Gauge: the first camera is fixed; after every accepted step each
point is rescaled to unit norm and each free camera to unit Frobenius norm
(scale gauges), so J^T J is rank-deficient only along the remaining PGL(4)
directions, which the LM damping handles.

d dehom(y)/dy = [[1/y2, 0, -y0/y2^2], [0, 1/y2, -y1/y2^2]];  dy/dP_k = X (row k),
dy/dX = P.
"""

from __future__ import annotations

import numpy as np


def _residuals(cameras: np.ndarray, X: np.ndarray, observations: np.ndarray) -> np.ndarray:
    y = np.einsum("vkj,ji->vik", cameras, X)                          # (V, N, 3)
    return (y[..., :2] / y[..., 2:3] - observations).ravel()


def _jacobian(cameras: np.ndarray, X: np.ndarray) -> np.ndarray:
    V, N = cameras.shape[0], X.shape[1]
    y = np.einsum("vkj,ji->vik", cameras, X)                          # (V, N, 3)
    inv = 1.0 / y[..., 2]
    D = np.zeros((V, N, 2, 3))
    D[..., 0, 0] = inv
    D[..., 1, 1] = inv
    D[..., 0, 2] = -y[..., 0] * inv ** 2
    D[..., 1, 2] = -y[..., 1] * inv ** 2
    n_cam = 12 * (V - 1)
    J = np.zeros((V, N, 2, n_cam + 4 * N))
    dcam = np.einsum("vnak,ni->vnaki", D, X.T).reshape(V, N, 2, 12)   # d r / d P_v (row-major)
    for v in range(1, V):
        J[v, :, :, 12 * (v - 1):12 * v] = dcam[v]
    dX = np.einsum("vnak,vkj->vnaj", D, cameras)                      # (V, N, 2, 4)
    for i in range(N):
        J[:, i, :, n_cam + 4 * i:n_cam + 4 * i + 4] = dX[:, i]
    return J.reshape(2 * V * N, n_cam + 4 * N)


def _normalize(cameras: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = X / np.linalg.norm(X, axis=0, keepdims=True)
    cameras = cameras.copy()
    cameras[1:] /= np.linalg.norm(cameras[1:], axis=(1, 2), keepdims=True)
    return cameras, X


def bundle_adjust_lm(cameras: list[np.ndarray], X: np.ndarray, observations: list[np.ndarray],
                     max_iterations: int = 200, tolerance: float = 1e-12) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Returns cameras, unit points (4, N) and the sum of squared residuals (normalized units)."""
    cams = np.asarray(cameras, dtype=float)
    obs = np.asarray(observations, dtype=float)
    cams, X = _normalize(cams, np.asarray(X, dtype=float))
    V, N = cams.shape[0], X.shape[1]
    n_cam = 12 * (V - 1)
    r = _residuals(cams, X, obs)
    cost = float(r @ r)
    damping = 1e-3
    for _ in range(max_iterations):
        J = _jacobian(cams, X)
        JtJ, Jtr = J.T @ J, J.T @ r
        improved = False
        while damping < 1e12:
            A = JtJ + damping * np.diag(np.maximum(np.diag(JtJ), 1e-12))
            try:
                step = -np.linalg.solve(A, Jtr)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            new_cams = cams.copy()
            new_cams[1:] += step[:n_cam].reshape(V - 1, 3, 4)
            new_X = X + step[n_cam:].reshape(N, 4).T
            new_cams, new_X = _normalize(new_cams, new_X)
            new_r = _residuals(new_cams, new_X, obs)
            new_cost = float(new_r @ new_r) if np.all(np.isfinite(new_r)) else np.inf
            if new_cost < cost:
                relative = (cost - new_cost) / max(cost, 1e-300)
                cams, X, r, cost = new_cams, new_X, new_r, new_cost
                damping = max(damping / 10.0, 1e-12)
                improved = True
                break
            damping *= 10.0
        if not improved or relative < tolerance:
            break
    return list(cams), X, cost
