"""
forensics/geometry/brackets.py

Brackets in P^(d-1) for d in {3, 4} and the unified Plucker identity.

    d = 3: [x_i x_j x_k] = det of three image points (collinearity) or of three
           image lines (concurrency);
    d = 4: [p_i p_j p_k p_l] = det of four points of P^3 (coplanarity).

Unified Plucker identity [proved]: for d >= 2, W = (w_1..w_{d-2}) and any
x1..x4 in R^d,
    [W x1 x2][W x3 x4] - [W x1 x3][W x2 x4] + [W x1 x4][W x2 x3] = 0.
It holds for ANY points, so on its own it carries no information about
authenticity (for d = 4 see docs/MATHEMATICS.md, Corollary 2.6). Only
constraints that encode a scene hypothesis (collinearity of a straight edge,
a common vanishing point, rigidity across views) can detect anything.
"""

from __future__ import annotations
from fractions import Fraction
from itertools import combinations

import numpy as np


def bracket(*vectors: np.ndarray) -> float:
    """det of d vectors in R^d (d = number of vectors)."""
    matrix = np.column_stack(vectors)
    assert matrix.shape[0] == matrix.shape[1], "need d vectors of length d"
    return float(np.linalg.det(matrix))


def exact_integer_determinant(matrix: list[list[int]]) -> int:
    """Exact determinant of an integer matrix by fraction-free Gaussian elimination."""
    m = [[Fraction(v) for v in row] for row in matrix]
    n = len(m)
    det = Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if m[r][col] != 0), None)
        if pivot is None:
            return 0
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
            det = -det
        det *= m[col][col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n):
                m[r][c] -= factor * m[col][c]
    assert det.denominator == 1
    return int(det)


def plucker_identity_residual(W: list[np.ndarray], x1, x2, x3, x4) -> float:
    """Left side of the unified Plucker identity (zero for all inputs, up to rounding)."""
    def b(x, y):
        return bracket(*W, x, y)
    return b(x1, x2) * b(x3, x4) - b(x1, x3) * b(x2, x4) + b(x1, x4) * b(x2, x3)


def exact_bracket_table(points: list[list[int]]) -> dict[tuple[int, int, int, int], int]:
    """All 15 brackets of six integer points of P^3 (points[i] has length 4), exactly."""
    assert len(points) == 6
    return {
        quad: exact_integer_determinant([[points[i][r] for i in quad] for r in range(4)])
        for quad in combinations(range(6), 4)
    }


def signed_bracket(table: dict, i: int, j: int, k: int, l: int):
    """Look up [ijkl] in a table keyed by sorted quads, with the permutation sign."""
    order = [i, j, k, l]
    inversions = sum(1 for a in range(4) for b in range(a + 1, 4) if order[a] > order[b])
    return (-1) ** inversions * table[tuple(sorted(order))]


def fifteen_plucker_relations(table: dict) -> list:
    """Value of each three-term relation [abcd][abef] - [abce][abdf] + [abcf][abde]."""
    values = []
    for a, b in combinations(range(6), 2):
        c, d, e, f = [i for i in range(6) if i not in (a, b)]
        B = lambda *idx: signed_bracket(table, *idx)  # noqa: E731
        values.append(B(a, b, c, d) * B(a, b, e, f) - B(a, b, c, e) * B(a, b, d, f)
                      + B(a, b, c, f) * B(a, b, d, e))
    return values


def manuscript_sigmas(table: dict) -> tuple:
    """Sigma_1, Sigma_2, Sigma_3, the relations behind rho (0-based indices)."""
    B = lambda *idx: signed_bracket(table, *idx)  # noqa: E731
    sigma_1 = B(0, 1, 2, 3) * B(0, 1, 4, 5) - B(0, 1, 2, 4) * B(0, 1, 3, 5) + B(0, 1, 2, 5) * B(0, 1, 3, 4)
    sigma_2 = B(0, 2, 3, 4) * B(1, 2, 3, 5) - B(0, 2, 3, 5) * B(1, 2, 3, 4) - B(0, 1, 2, 3) * B(2, 3, 4, 5)
    sigma_3 = B(0, 1, 4, 5) * B(2, 3, 4, 5) - B(0, 2, 4, 5) * B(1, 3, 4, 5) + B(0, 3, 4, 5) * B(1, 2, 4, 5)
    return sigma_1, sigma_2, sigma_3


def b_star_table() -> dict[tuple[int, int, int, int], int]:
    """
    Counterexample of docs/MATHEMATICS.md, Proposition 6.2: b* = e_1234 + e_1456 (1-based), all other
    brackets 0. Sigma_1 = Sigma_2 = Sigma_3 = 0, yet the {1,4} relation equals 1,
    so b* is realized by no six points.
    """
    table = {quad: 0 for quad in combinations(range(6), 4)}
    table[(0, 1, 2, 3)] = 1
    table[(0, 3, 4, 5)] = 1
    return table
