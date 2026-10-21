"""
algorithms/consistency/pencil.py

Pencil cross-ratios of a point set in P^3 and the pencil drift test
(docs/MATHEMATICS.md, Sections 2-4).

For points p_1..p_n (columns of a 4 x n array), a line (i j) and an ordered
quadruple (c, d, e, f) of further points, the pencil cross-ratio is

    CR_ij(c, d; e, f) = [i j c e][i j d f] / ([i j c f][i j d e]),

the cross-ratio of the four planes span(p_i, p_j, p_x), x in {c, d, e, f},
of the pencil through the line p_i p_j (Lemma 2.2). It is invariant under
p_k -> lambda_k A p_k for any A in GL(4) and nonzero lambda_k (Lemma 2.1).
The six-point invariants J1, J2, J3 are pencil cross-ratios (Proposition 2.3).

For every line and every 4-set of the remaining points three pairings are
used, ((c,d),(e,f)), ((c,e),(d,f)), ((c,f),(d,e)), giving
C(n,2) * C(n-2,4) * 3 values (41,580 for n = 12). Each log|CR| is a signed
sum of four of the C(n,4) log-brackets, so all of them cost one batch of
C(n,4) determinants.

Drift test (Section 4): for two measurements A, B of the same points, in
arbitrary projective frames,

    D_t = | log|CR_t(A)| - log|CR_t(B)| |,     score = Q_p({D_t}) / delta,

with p = 0.1. To first order the score is (r / delta) times a variable that
does not depend on r or delta (Theorem 4.1). For small p the quantile follows
the harmonic mean of the per-invariant noise scales (Proposition 4.3), so it
is governed by well-conditioned invariants.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

from algorithms.backend import get_cpp_module

DEFAULT_QUANTILE = 0.1
# Above this family size the pencils are sampled uniformly (with a fixed seed):
# the full family has C(n,2) C(n-2,4) * 3 members, which grows like n^6
# (41,580 at n = 12, 77,220 at n = 13, 1.74 million at n = 20).
DEFAULT_MAX_PENCILS = 50_000
# Pairings of a 4-set {x0, x1, x2, x3} as ((c, d), (e, f)): CR = [ce][df] / ([cf][de]).
PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


@lru_cache(maxsize=16)
def pencil_index(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (quads, terms):
      quads (C(n,4), 4) sorted point quadruples;
      terms (T, 4) indices into quads: log|CR_t| = L[t0] + L[t1] - L[t2] - L[t3].
    Signs of the brackets do not matter for log|CR|.
    """
    assert n_points >= 6
    quads = list(combinations(range(n_points), 4))
    index = {q: k for k, q in enumerate(quads)}

    def q(*points):
        return index[tuple(sorted(points))]

    terms = []
    for i, j in combinations(range(n_points), 2):
        rest = [k for k in range(n_points) if k not in (i, j)]
        for four in combinations(rest, 4):
            for (a1, a2), (b1, b2) in PAIRINGS:
                c, d, e, f = four[a1], four[a2], four[b1], four[b2]
                # CR_ij(c, d; e, f) = [ijce][ijdf] / ([ijcf][ijde])
                terms.append((q(i, j, c, e), q(i, j, d, f), q(i, j, c, f), q(i, j, d, e)))
    return np.array(quads, dtype=int), np.array(terms, dtype=int)


def pencil_family_size(n_points: int) -> int:
    return comb(n_points, 2) * comb(n_points - 2, 4) * 3


def log_abs_brackets(points_homogeneous: np.ndarray, quads: np.ndarray | None = None) -> np.ndarray:
    """
    log|det| of the unit-normalized columns of each quadruple (default: every
    sorted quadruple, C(n,4) of them). Uses the C++ kernel when built.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    if quads is None:
        quads, _ = pencil_index(points_homogeneous.shape[1])
    quads = np.ascontiguousarray(quads, dtype=np.int64)
    cpp = get_cpp_module()
    if cpp is not None and hasattr(cpp, "log_abs_brackets"):
        return np.asarray(cpp.log_abs_brackets(np.ascontiguousarray(points_homogeneous, dtype=float), quads))
    unit = points_homogeneous / np.linalg.norm(points_homogeneous, axis=0, keepdims=True)
    return np.linalg.slogdet(np.transpose(unit[:, quads], (1, 0, 2)))[1]


def sample_pencil_quads(n_points: int, n_pencils: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform sample (with replacement) of pencil cross-ratios: a random line
    (i j), a random ordered 4-set (c, d, e, f) of further points, which with
    the three pairings is uniform over the family. Returns (K, 4, 4) point
    quadruples for the four brackets [ijce], [ijdf], [ijcf], [ijde].
    """
    # Rejection sampling of 6 distinct points per row: uniform over ordered
    # 6-tuples, O(K) work instead of the O(K n log n) of a per-row permutation.
    assert n_points >= 6
    six = rng.integers(0, n_points, size=(n_pencils, 6))
    pending = np.arange(n_pencils)                    # rows still to be checked
    while pending.size:
        rows = np.sort(six[pending], axis=1)
        pending = pending[(rows[:, 1:] == rows[:, :-1]).any(axis=1)]
        six[pending] = rng.integers(0, n_points, size=(pending.size, 6))
    i, j, c, d, e, f = six.T
    return np.stack([np.stack([i, j, c, e], 1), np.stack([i, j, d, f], 1),
                     np.stack([i, j, c, f], 1), np.stack([i, j, d, e], 1)], axis=1)


@lru_cache(maxsize=32)
def _sampled_quads(n_points: int, n_pencils: int, seed: int) -> np.ndarray:
    """The uniform pencil sample of sample_pencil_quads for (n, K, seed), flattened to (4K, 4) and cached
    (read-only): windows or pairs that reuse a seed skip the resampling. Identical to sampling afresh."""
    quads = sample_pencil_quads(n_points, n_pencils, np.random.default_rng(seed)).reshape(-1, 4)
    quads.setflags(write=False)
    return quads


def log_abs_pencil_cross_ratios(points_homogeneous: np.ndarray) -> np.ndarray:
    """log|CR_t| for every pencil cross-ratio t (gauge-invariant)."""
    _, terms = pencil_index(points_homogeneous.shape[1])
    L = log_abs_brackets(points_homogeneous)
    with np.errstate(invalid="ignore"):
        return L[terms[:, 0]] + L[terms[:, 1]] - L[terms[:, 2]] - L[terms[:, 3]]


def pencil_discrepancies(points_a: np.ndarray, points_b: np.ndarray, max_pencils: int = DEFAULT_MAX_PENCILS,
                         seed: int = 0) -> np.ndarray:
    """
    D_t = |log|CR_t(A)| - log|CR_t(B)|| for two measurements of the same points
    (any frames). The full family is used when it has at most max_pencils
    members; otherwise max_pencils pencils are sampled uniformly with the given
    seed, and Q_p of the sample estimates Q_p of the family.
    """
    assert points_a.shape == points_b.shape
    n = points_a.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):      # a zero bracket gives inf / NaN entries
        if pencil_family_size(n) <= max_pencils:
            return np.abs(log_abs_pencil_cross_ratios(points_a) - log_abs_pencil_cross_ratios(points_b))
        quads = _sampled_quads(n, max_pencils, seed)
        la = log_abs_brackets(points_a, quads).reshape(-1, 4)
        lb = log_abs_brackets(points_b, quads).reshape(-1, 4)
        diff = la - lb
        return np.abs(diff[:, 0] + diff[:, 1] - diff[:, 2] - diff[:, 3])


def pencil_drift_score(points_a: np.ndarray, points_b: np.ndarray, deformation_bound: float = 1.0,
                       quantile: float = DEFAULT_QUANTILE, max_pencils: int = DEFAULT_MAX_PENCILS,
                       seed: int = 0) -> float:
    """
    Q_p of the pencil discrepancies divided by the admissible deformation bound.
    Larger means more deformation than the bound allows; the decision threshold
    is calibrated on authentic or bounded data (Theorem 4.1 makes it independent
    of the bound to first order). Exact for n <= 12 points (41,580 pencils);
    sampled (50,000 pencils, fixed seed) above that.
    """
    assert deformation_bound > 0 and 0 < quantile < 1
    D = pencil_discrepancies(points_a, points_b, max_pencils, seed)
    D = D[np.isfinite(D)]           # pencils through a numerically zero bracket carry no information
    return float(np.quantile(D, quantile)) / deformation_bound if D.size else float("nan")


def log_abs_pencil_gradients(points_homogeneous: np.ndarray) -> np.ndarray:
    """
    Gradients of log|CR_t| with respect to the 4n coordinates (point-major:
    entry 4*k + r is coordinate r of point k), evaluated at the given
    representatives (Lemma 3.1: d log|det M| = tr(M^{-1} dM), so the gradient
    with respect to column m of M is row m of M^{-1}). Shape (T, 4n).
    """
    n = points_homogeneous.shape[1]
    quads, terms = pencil_index(n)
    M = np.transpose(points_homogeneous[:, quads], (1, 0, 2))          # (Q, 4, 4)
    inv = np.linalg.inv(M)                                             # rows = gradients per column
    G = np.zeros((len(quads), 4 * n))
    for m in range(4):
        cols = (4 * quads[:, m])[:, None] + np.arange(4)[None, :]
        np.put_along_axis(G, cols, inv[:, m, :], axis=1)
    return G[terms[:, 0]] + G[terms[:, 1]] - G[terms[:, 2]] - G[terms[:, 3]]


def pencil_point_scores(points_a: np.ndarray, points_b: np.ndarray, max_pencils: int = DEFAULT_MAX_PENCILS,
                        seed: int = 0) -> np.ndarray:
    """
    Per-point localisation heuristic: for each point, the median pencil
    discrepancy over the sampled pencils that involve it. A point that is
    mismatched between A and B raises every pencil it enters, so its median
    rises. This is a heuristic (docs/MATHEMATICS.md proves the scene-level test,
    not this per-point rule); in the outlier benchmark the analogous per-point
    median of Delta_J reached AUC 0.966 against 1.0 for RANSAC registration.
    Returns (n,) scores (NaN for a point in no sampled pencil).
    """
    assert points_a.shape == points_b.shape
    n = points_a.shape[1]
    n_pencils = min(max_pencils, pencil_family_size(n))
    rng = np.random.default_rng(seed)
    quads = sample_pencil_quads(n, n_pencils, rng)                        # (K, 4, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        la = log_abs_brackets(points_a, quads.reshape(-1, 4)).reshape(-1, 4)
        lb = log_abs_brackets(points_b, quads.reshape(-1, 4)).reshape(-1, 4)
        diff = la - lb
        D = np.abs(diff[:, 0] + diff[:, 1] - diff[:, 2] - diff[:, 3])
    finite = np.isfinite(D)                       # a numerically zero bracket gives inf / NaN: ignore it
    six = np.concatenate([quads[:, 0, :], quads[:, 1, 2:]], axis=1)[finite]   # i, j, c, e, d, f
    D = D[finite]
    scores = np.full(n, np.nan)
    for point in range(n):
        involved = np.any(six == point, axis=1)
        if involved.any():
            scores[point] = float(np.median(D[involved]))
    return scores
