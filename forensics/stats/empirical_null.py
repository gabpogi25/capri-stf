"""
forensics/stats/empirical_null.py

Empirical null calibration (the real-media null gate).

The chi^2 null of nulls.py rests on a noise model that real edge pixels
and tracks need not follow. Before a threshold is used, the statistic is
calibrated on held-out AUTHENTIC media: p-values are the upper-tail ranks of
a test statistic within that reference sample. the real-media null gate then checks that the
rejection rate at nominal level alpha on a further authentic split falls in
the pre-registered band.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class EmpiricalNull:
    reference: np.ndarray            # sorted statistics of authentic calibration media

    @classmethod
    def fit(cls, authentic_statistics: np.ndarray) -> "EmpiricalNull":
        values = np.asarray(authentic_statistics, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 20:
            raise ValueError(f"Empirical null needs at least 20 authentic values, got {values.size}")
        return cls(np.sort(values))

    def p_value(self, statistic: float | np.ndarray) -> np.ndarray:
        """(1 + #{reference >= s}) / (n + 1): valid, conservative upper-tail p-value."""
        s = np.atleast_1d(np.asarray(statistic, dtype=float))
        n = self.reference.size
        greater_equal = n - np.searchsorted(self.reference, s, side="left")
        return (1.0 + greater_equal) / (n + 1.0)

    def threshold(self, alpha: float) -> float:
        return float(np.quantile(self.reference, 1.0 - alpha))


def rejection_rate(p_values: np.ndarray, alpha: float) -> float:
    p = np.asarray(p_values, dtype=float)
    return float(np.mean(p <= alpha))


def uniformity_ks(p_values: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov distance of p-values to U(0, 1) and its p-value (QQ diagnostic)."""
    result = stats.kstest(np.asarray(p_values, dtype=float), "uniform")
    return float(result.statistic), float(result.pvalue)
