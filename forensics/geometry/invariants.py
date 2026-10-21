"""
forensics/geometry/invariants.py

Six-point invariants J1, J2, J3 of P^3, their logarithms, conditioning and
noise bounds (tau_exact is Theorem 6.1 of docs/MATHEMATICS.md).

Column order of the 4 x 6 matrix P is p1..p6:
    J1 = [1234][1256] / ([1235][1246])
    J2 = [1345][2346] / ([1346][2345])
    J3 = [1256][3456] / ([1356][2456])
Each point appears with equal weight in numerator and denominator, so every
J_i is invariant under p_i -> lambda_i p_i and under P -> A P, A in GL(4).
"""

from __future__ import annotations
import numpy as np

_J_BRACKETS = (
    # (numerator quads, denominator quads), 0-based
    (((0, 1, 2, 3), (0, 1, 4, 5)), ((0, 1, 2, 4), (0, 1, 3, 5))),
    (((0, 2, 3, 4), (1, 2, 3, 5)), ((0, 2, 3, 5), (1, 2, 3, 4))),
    (((0, 1, 4, 5), (2, 3, 4, 5)), ((0, 2, 4, 5), (1, 3, 4, 5))),
)


def _det(P: np.ndarray, quad: tuple[int, int, int, int]) -> float:
    return float(np.linalg.det(P[:, list(quad)]))


def invariants(P: np.ndarray) -> np.ndarray:
    """(J1, J2, J3) of a 4 x 6 matrix; NaN where a denominator vanishes exactly."""
    assert P.shape == (4, 6)
    values = []
    for (n1, n2), (d1, d2) in _J_BRACKETS:
        denominator = _det(P, d1) * _det(P, d2)
        values.append(_det(P, n1) * _det(P, n2) / denominator if denominator != 0 else np.nan)
    return np.asarray(values)


def log_abs_invariants(P: np.ndarray) -> np.ndarray:
    """log |J_i| computed as sums/differences of log |brackets| (more stable than log of a ratio)."""
    assert P.shape == (4, 6)
    out = []
    for (n1, n2), (d1, d2) in _J_BRACKETS:
        logs = [np.log(abs(_det(P, q))) if _det(P, q) != 0 else -np.inf for q in (n1, n2, d1, d2)]
        out.append(logs[0] + logs[1] - logs[2] - logs[3])
    return np.asarray(out)


def log_invariant_jacobian(P: np.ndarray, step: float | None = None) -> np.ndarray:
    """
    3 x 24 Jacobian of log|J| with respect to the entries of P (column-major =
    point-major), exact: d log|det M| = tr(M^{-1} dM) (docs/MATHEMATICS.md,
    Lemma 3.1); central differences would be about 40x slower and accurate only
    to ~1e-9. `step` is accepted for interface compatibility and ignored.
    """
    from algorithms.conditioning.scores import log_invariant_gradients

    assert P.shape == (4, 6)
    return log_invariant_gradients(np.asarray(P, dtype=float), tuple(range(6)), normalize=False)


def kappa_j(P: np.ndarray, sigma0_sqrt: np.ndarray | None = None) -> float:
    """
    Chart-free conditioning: max_i || Sigma_0^{1/2} grad log J_i ||.

    sigma0_sqrt is the 24 x 24 square root of the measurement covariance of
    the entries of P (column-major); identity if omitted. Large kappa means the
    log-invariants amplify measurement noise; such tuples are gated out.
    """
    jacobian = log_invariant_jacobian(P)
    if sigma0_sqrt is not None:
        jacobian = jacobian @ sigma0_sqrt
    norms = np.linalg.norm(jacobian, axis=1)
    return float(np.max(norms)) if np.all(np.isfinite(norms)) else float("inf")


def j1_delta_min(P_unit: np.ndarray) -> float:
    """Smallest |bracket| among the four brackets of J1 (unit-norm columns)."""
    (n1, n2), (d1, d2) = _J_BRACKETS[0]
    return float(min(abs(_det(P_unit, q)) for q in (n1, n2, d1, d2)))


def amplification_bound(sigma: float, delta_min: float) -> float:
    """
    Noise bound: |Delta log J1| <= 32 sigma (1 + sigma)^3 / delta_min,
    valid when every unit column moves by at most sigma and
    4 sigma (1 + sigma)^3 <= delta_min / 2. Returns inf outside that regime.
    """
    if 4 * sigma * (1 + sigma) ** 3 > delta_min / 2:
        return float("inf")
    return 32.0 * sigma * (1 + sigma) ** 3 / delta_min


def tau_exact(delta: float) -> float:
    """Theorem 6.1: rho <= 3 eps (2 + eps), eps = (1 + delta)^4 - 1."""
    eps = (1.0 + delta) ** 4 - 1.0
    return 3.0 * eps * (2.0 + eps)


def sigma4(P: np.ndarray) -> float:
    """Smallest singular value of the column-normalized 4 x m matrix (distance to coplanarity)."""
    unit = P / np.linalg.norm(P, axis=0, keepdims=True)
    return float(np.linalg.svd(unit, compute_uv=False)[3])
