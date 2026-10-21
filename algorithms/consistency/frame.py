"""
algorithms/consistency/frame.py

Frame-coordinate consistency statistic (docs/MATHEMATICS.md, Proposition 3.5 and Theorem 3.7).

Five points f1..f5 in general position form a projective frame. For every other point k,
the three pencil cross-ratios

    CR_{f1 f2}(f3, f4; f5, k),   CR_{f1 f3}(f2, f4; f5, k),   CR_{f2 f3}(f1, f4; f5, k)

are the ratios z/w, y/w, x/w of the frame coordinates of point k. The 3(n - 5) of them
determine the configuration up to the gauge (signed; Corollary 3.8) and have Jacobian
rank 3n - 15, the rank of the full pencil family (Theorem 3.7). They are gauge-invariant
(Lemma 2.1), so two measurements in different frames can be compared directly, with
O(n) brackets instead of the C(n,4) the full family needs.

The frame is chosen once for both measurements, to maximise the smallest of its own five
brackets and of the brackets [f_a f_b f_c k] that appear in the cross-ratios, taken over
both measurements (unit-normalised columns). This is a conditioning heuristic.
Status of the statistic as a test: exploratory (numerical); not a confirmatory result.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from algorithms.consistency.pencil import log_abs_brackets

# (i, j, c, d, e) of CR_ij(c, d; e, k) in frame positions 0..4: CR_12(3,4;5,k), CR_13(2,4;5,k), CR_23(1,4;5,k)
FRAME_PATTERN = ((0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (1, 2, 0, 3, 4))
MAX_FRAME_CANDIDATES = 2000


def _frame_quads(frame: tuple[int, ...], others: np.ndarray) -> np.ndarray:
    """(3 * len(others), 4, 4): the four bracket quadruples [ijce], [ijdf], [ijcf], [ijde] of each frame cross-ratio."""
    rows = []
    for k in others:
        for (i, j, c, d, e) in FRAME_PATTERN:
            I, J, C, D, E = (frame[t] for t in (i, j, c, d, e))
            rows.append(((I, J, C, E), (I, J, D, k), (I, J, C, k), (I, J, D, E)))
    return np.array(rows, dtype=int)


def frame_log_coordinates(points: np.ndarray, frame: tuple[int, ...]) -> np.ndarray:
    """log|CR| of the 3(n - 5) frame cross-ratios (point order: increasing k outside the frame)."""
    assert points.ndim == 2 and points.shape[0] == 4 and len(set(frame)) == 5
    others = np.array([k for k in range(points.shape[1]) if k not in frame])
    quads = _frame_quads(tuple(frame), others)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = log_abs_brackets(points, quads.reshape(-1, 4)).reshape(-1, 4)
    return L[:, 0] + L[:, 1] - L[:, 2] - L[:, 3]


def _cross4(M: np.ndarray) -> np.ndarray:
    """Generalised cross product of the three columns of each (.., 4, 3) matrix: det(a, b, c, x) = N . x."""
    minors = [np.linalg.det(np.delete(M, r, axis=-2)) for r in range(4)]
    return np.stack([(-1) ** (r + 1) * minors[r] for r in range(4)], axis=-1)


def choose_frame(points_a: np.ndarray, points_b: np.ndarray, seed: int = 0) -> tuple[int, ...]:
    """
    The 5-subset (in a fixed order) whose smallest |bracket| (unit columns) among the brackets it
    uses (its own five, and [f_a f_b f_c k] for the four trios of f1..f4 and every other k) is
    largest, over both measurements. Exhaustive when C(n,5) <= MAX_FRAME_CANDIDATES, otherwise
    that many 5-subsets drawn with the given seed. Each [f_a f_b f_c k] is computed as N . p_k.
    """
    n = points_a.shape[1]
    assert points_b.shape == points_a.shape and n >= 6
    if comb(n, 5) <= MAX_FRAME_CANDIDATES:
        subsets = list(combinations(range(n), 5))
    else:
        rng = np.random.default_rng(seed)
        subsets = [tuple(int(v) for v in sorted(rng.choice(n, 5, replace=False))) for _ in range(MAX_FRAME_CANDIDATES)]
    F = np.array(subsets)                                                        # (C, 5)
    trios = F[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]                   # (C, 4, 3)
    in_frame = np.zeros((len(F), n), dtype=bool)
    np.put_along_axis(in_frame, F, True, axis=1)
    worst = np.full(len(F), np.inf)
    for P in (points_a, points_b):
        U = P / np.linalg.norm(P, axis=0, keepdims=True)
        own = np.abs(np.linalg.det(np.transpose(U[:, F[:, list(combinations(range(5), 4))]], (1, 2, 0, 3))))  # (C, 5)
        N = _cross4(np.transpose(U[:, trios], (1, 2, 0, 3)))                    # (C, 4, 4)
        used = np.abs(N @ U)                                                      # (C, 4, n)
        used = np.where(in_frame[:, None, :], np.inf, used).min(axis=(1, 2))
        worst = np.minimum(worst, np.minimum(own.min(axis=1), used))
    return subsets[int(np.argmax(worst))]


def frame_discrepancies(points_a: np.ndarray, points_b: np.ndarray, frame: tuple[int, ...] | None = None,
                        seed: int = 0) -> np.ndarray:
    """|log|CR_t(A)| - log|CR_t(B)|| over the 3(n - 5) frame cross-ratios (frame chosen if not given)."""
    if frame is None:
        frame = choose_frame(points_a, points_b, seed)
    return np.abs(frame_log_coordinates(points_a, frame) - frame_log_coordinates(points_b, frame))


def frame_drift_score(points_a: np.ndarray, points_b: np.ndarray, deformation_bound: float = 1.0,
                      quantile: float = 0.1, frame: tuple[int, ...] | None = None, seed: int = 0) -> float:
    """Quantile of the frame discrepancies divided by the deformation bound (larger = more drift)."""
    assert deformation_bound > 0 and 0 < quantile < 1
    D = frame_discrepancies(points_a, points_b, frame, seed)
    D = D[np.isfinite(D)]
    return float(np.quantile(D, quantile)) / deformation_bound if D.size else float("nan")
