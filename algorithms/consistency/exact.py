"""
algorithms/consistency/exact.py

Exact brackets for floating-point input (plan item A6/B4). Every float is a
rational number, so det of four float columns has an exact rational value; computing it
with Fractions removes rounding error from decisions such as "is this bracket zero" or
"is |bracket| below tau". The value is exact for the floats given; it says nothing about
the measurement error those floats carry.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def _det4(M: list[list[Fraction]]) -> Fraction:
    def det3(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    return sum((-1) ** c * M[0][c] * det3([[M[r][k] for k in range(4) if k != c] for r in range(1, 4)])
               for c in range(4))


def exact_bracket(points: np.ndarray, cols: tuple[int, int, int, int]) -> Fraction:
    """Exact det(p_a, p_b, p_c, p_d) of the given float columns (no normalisation)."""
    assert points.shape[0] == 4 and len(cols) == 4
    return _det4([[Fraction(float(points[r, c])) for c in cols] for r in range(4)])


def bracket_below(points: np.ndarray, cols: tuple[int, int, int, int], tau: float) -> bool:
    """Certified |det| < tau * prod |p_k| (the unit-column bracket below tau), decided in exact arithmetic
    by comparing squares: det^2 < tau^2 prod |p_k|^2."""
    assert tau >= 0
    det = exact_bracket(points, cols)
    norms2 = [sum(Fraction(float(points[r, c])) ** 2 for r in range(4)) for c in cols]
    prod = norms2[0] * norms2[1] * norms2[2] * norms2[3]
    return det * det < Fraction(tau) ** 2 * prod
