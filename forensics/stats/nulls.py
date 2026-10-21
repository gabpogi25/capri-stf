"""
forensics/stats/nulls.py

Calibrated residual statistics.

For a constraint F with F(x) = 0 at the true measurements and measurement
noise e = sigma * Sigma_0^{1/2} z, the residual r = F(x_hat) has, to first
order, covariance Sigma_r = J Sigma_e J^T (J = dF/dx). The statistic
    T = r^T Sigma_r^+ r
converges to chi^2 with s = rank(Sigma_r) degrees of freedom as sigma -> 0.
The error of this first-order approximation is bounded by sigma * M * ||Sigma_0|| * ||z||^2 / (2 s_min),
so the chi^2 p-value is trustworthy only for well-conditioned constraints.
Real data must be checked against an empirical null before any threshold is
used (see empirical_null.py); chi^2 p-values are diagnostics.
"""

from __future__ import annotations
from typing import Callable

import numpy as np
from scipy import stats


def numerical_jacobian(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of a vector function."""
    x = np.asarray(x, dtype=float)
    f0 = np.atleast_1d(func(x))
    jacobian = np.zeros((f0.size, x.size))
    for k in range(x.size):
        h = step * max(1.0, abs(x[k]))
        plus, minus = x.copy(), x.copy()
        plus[k] += h
        minus[k] -= h
        jacobian[:, k] = (np.atleast_1d(func(plus)) - np.atleast_1d(func(minus))) / (2 * h)
    return jacobian


def delta_method_covariance(func, x: np.ndarray, measurement_covariance: np.ndarray) -> np.ndarray:
    """First-order covariance J Sigma J^T of func(x_hat)."""
    jacobian = numerical_jacobian(func, x)
    return jacobian @ measurement_covariance @ jacobian.T


def chi2_statistic(residual: np.ndarray, covariance: np.ndarray, rank_tol: float = 1e-10) -> tuple[float, int, float]:
    """
    T = r^T Sigma^+ r, its degrees of freedom s = rank(Sigma), and the
    chi^2_s upper-tail p-value.
    """
    residual = np.atleast_1d(np.asarray(residual, dtype=float))
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > rank_tol * max(eigenvalues.max(), 1e-300)
    projected = eigenvectors[:, keep].T @ residual
    statistic = float(np.sum(projected ** 2 / eigenvalues[keep]))
    dof = int(keep.sum())
    p_value = float(stats.chi2.sf(statistic, dof)) if dof > 0 else float("nan")
    return statistic, dof, p_value


def standardized_residual(residual: float, variance: float) -> float:
    """z = r / sqrt(var) for a scalar constraint."""
    return float(residual / np.sqrt(variance)) if variance > 0 else float("nan")
