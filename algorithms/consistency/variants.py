"""
algorithms/consistency/variants.py

Alternative consistency statistics for two measurements A, B of the same points
in arbitrary projective frames (docs/prereg/evaluation_plan.md, arms of clip aggregation/detector benchmark). They
are evaluated against the pencil drift statistic (pencil.py); none of them is
part of the original model and none is claimed to be new.

1. Signed pencil cross-ratios and the asinh transform.
   CR_t = [ijce][ijdf] / ([ijcf][ijde]) with SIGNED brackets is invariant under
   p_k -> l_k A p_k (each point and det A enter numerator and denominator equally
   often, so signs cancel too). asinh(CR) is therefore invariant as well; unlike
   log|CR| it is finite and smooth at CR = 0.  D_t = |asinh CR_t(A) - asinh CR_t(B)|.

2. Per-pencil first-order scale.  g_t = ||grad log|CR_t||| over the unit
   coordinates of the six points (Lemma 3.1: d log|det M| = tr(M^{-1} dM)), summed
   over A and B. z_t = D_t / (m0 g_t) is a standardized discrepancy for a noise
   scale m0 calibrated on consistent data.

3. Additive gauge fit on log-brackets.  For B = l_k A p_k (any A in GL(4)),
       log|[q](B)| - log|[q](A)| = log|det A| + sum_{k in q} log|l_k|      (exact),
   so the difference of the C(n,4) log-brackets is c + sum_{k in q} s_k with
   n + 1 unknowns; every pencil cross-ratio annihilates this model. The test fits
   c, s by Huber IRLS; the statistic is the 90% quantile of |r| (a minority of
   inconsistent points leaves the median at the noise level, so the median is
   reported only as a drift variant). Residuals per point
   (median over the quads containing it) give a localisation score.

The identity in 3 is checked in tests/test_eval_variants.py.
"""

from __future__ import annotations

import numpy as np

from algorithms.consistency.pencil import sample_pencil_quads

HUBER_K = 1.345
IRLS_ITERATIONS = 20


def signed_brackets(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """det of the unit-normalized columns of each quadruple (signed), (Q,)."""
    unit = points / np.linalg.norm(points, axis=0, keepdims=True)
    return np.linalg.det(np.transpose(unit[:, quads], (1, 0, 2)))


def sampled_pencils(n_points: int, n_pencils: int, seed: int) -> np.ndarray:
    """(K, 4, 4) quadruples of K uniformly sampled pencils (same sampler as pencil.py)."""
    return sample_pencil_quads(n_points, n_pencils, np.random.default_rng(seed))


def signed_cross_ratios(points: np.ndarray, pencils: np.ndarray) -> np.ndarray:
    b = signed_brackets(points, pencils.reshape(-1, 4)).reshape(-1, 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        return b[:, 0] * b[:, 1] / (b[:, 2] * b[:, 3])


def asinh_discrepancies(points_a: np.ndarray, points_b: np.ndarray, pencils: np.ndarray) -> np.ndarray:
    return np.abs(np.arcsinh(signed_cross_ratios(points_a, pencils)) - np.arcsinh(signed_cross_ratios(points_b, pencils)))


def log_discrepancies(points_a: np.ndarray, points_b: np.ndarray, pencils: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(np.log(np.abs(signed_cross_ratios(points_a, pencils)))
                      - np.log(np.abs(signed_cross_ratios(points_b, pencils))))


def pencil_gradient_norms(points: np.ndarray, pencils: np.ndarray) -> np.ndarray:
    """||grad log|CR_t||| w.r.t. the unit coordinates of its six points, (K,). Bracket order per
    pencil: [ijce], [ijdf], [ijcf], [ijde] (sample_pencil_quads); log|CR| = L0 + L1 - L2 - L3."""
    unit = points / np.linalg.norm(points, axis=0, keepdims=True)
    M = np.transpose(unit[:, pencils.reshape(-1, 4)], (1, 0, 2)).reshape(-1, 4, 4, 4)   # (K, bracket, 4, 4)
    with np.errstate(all="ignore"):
        inv = np.linalg.inv(M)                                                          # rows = d log|det| / d column
    g0, g1, g2, g3 = inv[:, 0], inv[:, 1], inv[:, 2], inv[:, 3]                         # (K, 4 rows, 4)
    grads = [g0[:, 0] + g1[:, 0] - g2[:, 0] - g3[:, 0],     # i
             g0[:, 1] + g1[:, 1] - g2[:, 1] - g3[:, 1],     # j
             g0[:, 2] - g2[:, 2],                           # c
             g1[:, 2] - g3[:, 2],                           # d
             g0[:, 3] - g3[:, 3],                           # e
             g1[:, 3] - g2[:, 3]]                           # f
    return np.sqrt(sum(np.sum(g ** 2, axis=1) for g in grads))


def standardized_discrepancies(points_a, points_b, pencils, noise_scale: float) -> np.ndarray:
    """z_t = D_t / (m0 (g_t(A) + g_t(B))); NaN where a bracket is singular."""
    assert noise_scale > 0
    D = log_discrepancies(points_a, points_b, pencils)
    g = pencil_gradient_norms(points_a, pencils) + pencil_gradient_norms(points_b, pencils)
    with np.errstate(all="ignore"):
        return D / (noise_scale * g)


def gauge_fit(points_a: np.ndarray, points_b: np.ndarray, quads: np.ndarray) -> dict:
    """
    Huber IRLS fit of d_q = log|[q](B)| - log|[q](A)| = c + sum_{k in q} s_k + r_q over the given
    quadruples. Returns {"statistic": Q_0.9 |r| over finite rows, "median": median |r|, "point_scores": (n,) median |r|
    over the rows containing the point, "residuals": r}. The model has a one-dimensional
    non-identifiability (c -> c + 4a, s_k -> s_k - a); the minimum-norm solution is used.
    """
    n = points_a.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (np.log(np.abs(signed_brackets(points_b, quads))) - np.log(np.abs(signed_brackets(points_a, quads))))
    keep = np.isfinite(d)
    q, d = quads[keep], d[keep]
    weights = np.ones(len(d))
    residuals = d - np.median(d)
    flat = q.ravel()
    pair_flat = (q[:, :, None] * n + q[:, None, :]).ravel()
    for _ in range(IRLS_ITERATIONS):
        normal = np.zeros((n + 1, n + 1))
        rhs = np.zeros(n + 1)
        normal[0, 0] = weights.sum()
        rhs[0] = weights @ d
        point_weight = np.bincount(flat, np.repeat(weights, 4), minlength=n)
        normal[0, 1:] = normal[1:, 0] = point_weight
        rhs[1:] = np.bincount(flat, np.repeat(weights * d, 4), minlength=n)
        normal[1:, 1:] = np.bincount(pair_flat, np.repeat(weights, 16), minlength=n * n).reshape(n, n)
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        new_residuals = d - coefficients[0] - coefficients[1:][q].sum(axis=1)
        converged = np.max(np.abs(new_residuals - residuals)) < 1e-10
        residuals = new_residuals
        if converged:
            break
        scale = 1.4826 * np.median(np.abs(residuals)) + 1e-300
        u = np.abs(residuals) / (HUBER_K * scale)
        weights = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-300))
    point_scores = np.full(n, np.nan)
    absolute = np.abs(residuals)
    for k in range(n):
        rows = np.any(q == k, axis=1)
        if rows.any():
            point_scores[k] = np.median(absolute[rows])
    return {"statistic": float(np.quantile(absolute, 0.9)) if absolute.size else float("nan"),
            "median": float(np.median(absolute)) if absolute.size else float("nan"),
            "point_scores": point_scores, "residuals": residuals}
