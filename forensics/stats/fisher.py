"""
forensics/stats/fisher.py

Aggregation of per-constraint evidence.

Independent p-values combine by Fisher's method, -2 sum log p ~ chi^2_{2K}.
Per-frame and per-window evidence in a video is serially correlated, so the
clip statistic uses an effective sample size
    n_eff = n / (1 + 2 sum_{k>=1} rho_k),
with the autocorrelation sum truncated at the first lag whose autocorrelation
is not significant (|rho_k| < 2 / sqrt(n)).
"""

from __future__ import annotations
import numpy as np
from scipy import stats


def fisher_combine(p_values: np.ndarray) -> tuple[float, float]:
    """(statistic, combined p) for independent p-values."""
    p = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan"), float("nan")
    statistic = float(-2.0 * np.sum(np.log(p)))
    return statistic, float(stats.chi2.sf(statistic, 2 * p.size))


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator == 0:
        return np.zeros(max_lag)
    return np.array([np.dot(x[:-k], x[k:]) / denominator for k in range(1, max_lag + 1)])


def effective_sample_size(series: np.ndarray) -> float:
    """n / (1 + 2 sum rho_k), truncated at the first non-significant lag; at most n, at least 1."""
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 3:
        return float(n)
    rho = autocorrelation(x, min(n - 1, 50))
    threshold = 2.0 / np.sqrt(n)
    total = 0.0
    for value in rho:
        if abs(value) < threshold:
            break
        total += value
    return float(np.clip(n / (1.0 + 2.0 * total), 1.0, n))


def block_fisher(p_values: np.ndarray) -> tuple[float, float, float]:
    """
    Fisher combination with the effective sample size: the Fisher statistic is
    rescaled by n_eff / n and referred to chi^2 with 2 n_eff degrees of freedom.
    Returns (statistic, p, n_eff).
    """
    p = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan"), float("nan"), 0.0
    z = stats.norm.isf(p)
    n_eff = effective_sample_size(z)
    statistic = float(-2.0 * np.sum(np.log(p)) * n_eff / p.size)
    return statistic, float(stats.chi2.sf(statistic, 2 * n_eff)), n_eff
