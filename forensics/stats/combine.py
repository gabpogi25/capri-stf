"""
forensics/stats/combine.py

Combining p-values and e-values across pencils, tuples or windows
(docs/prereg/evaluation_plan.md, arms of clip aggregation and representation robustness check; standard methods, cited):

- Cauchy combination test (ACAT; Liu & Xie 2020): T = mean tan((0.5 - p_i) pi);
  p = 1/2 - arctan(T)/pi. Asymptotically valid under arbitrary dependence for
  small significance levels.
- Harmonic mean p-value (Wilson 2019), reported raw (1 / mean(1/p)); its
  asymptotically exact version needs the Landau law and is not used as a test.
- Bonferroni minimum: min(1, k min p).
- e-values (Vovk & Wang 2021): the calibrator e = kappa p^(kappa - 1), kappa in
  (0, 1), turns a valid p-value into an e-value; the product of independent
  e-values is an e-value, and rejecting when it reaches 1/alpha has level alpha
  (Markov), also under optional stopping when the windows arrive in sequence
  and each p-value is valid conditionally on the past.
"""

from __future__ import annotations

import numpy as np

E_CALIBRATOR_KAPPA = 0.5


def cauchy_combination(p_values: np.ndarray) -> float:
    p = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")
    terms = np.where(p < 1e-15, 1.0 / (p * np.pi), np.tan((0.5 - p) * np.pi))
    statistic = float(np.mean(terms))
    return float(0.5 - np.arctan(statistic) / np.pi) if statistic < 1e15 else float(1.0 / (statistic * np.pi))


def harmonic_mean_p(p_values: np.ndarray) -> float:
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]
    return float(1.0 / np.mean(1.0 / np.clip(p, 1e-300, 1.0))) if p.size else float("nan")


def bonferroni_min(p_values: np.ndarray) -> float:
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]
    return float(min(1.0, p.size * p.min())) if p.size else float("nan")


def p_to_e(p_values: np.ndarray, kappa: float = E_CALIBRATOR_KAPPA) -> np.ndarray:
    assert 0 < kappa < 1
    return kappa * np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0) ** (kappa - 1.0)


def e_product(p_values: np.ndarray, kappa: float = E_CALIBRATOR_KAPPA) -> float:
    """Product of calibrated e-values (log-sum for stability). Reject at level alpha iff >= 1/alpha."""
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]
    return float(np.exp(np.sum(np.log(p_to_e(p, kappa))))) if p.size else float("nan")


def e_process_alarm(p_values: np.ndarray, alpha: float, kappa: float = E_CALIBRATOR_KAPPA) -> int | None:
    """First index at which the running product of e-values reaches 1/alpha (anytime-valid), else None."""
    running = np.cumsum(np.log(p_to_e(np.asarray(p_values, dtype=float), kappa)))
    hits = np.flatnonzero(running >= np.log(1.0 / alpha))
    return int(hits[0]) if hits.size else None
