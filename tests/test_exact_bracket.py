"""Exact brackets for float input (algorithms/consistency/exact.py)."""

from fractions import Fraction

import numpy as np

from algorithms.consistency.exact import bracket_below, exact_bracket


def test_exact_zero_where_floating_point_is_not():
    rng = np.random.default_rng(10)
    a, b, c = (rng.integers(-10 ** 6, 10 ** 6, 4).astype(float) for _ in range(3))
    k = rng.integers(-50, 50, 3).astype(float)
    P = np.column_stack([a * 0.5 ** 20, b, c, k[0] * a + k[1] * b + k[2] * c])   # exactly dependent columns
    assert np.linalg.det(P) != 0                                                  # floating point misses it
    assert exact_bracket(P, (0, 1, 2, 3)) == 0
    assert bracket_below(P, (0, 1, 2, 3), 0.0) is False   # |det| < 0 is false even for an exact zero
    assert bracket_below(P, (0, 1, 2, 3), 1e-300) is True


def test_agrees_with_numpy_on_generic_input():
    rng = np.random.default_rng(9)
    P = rng.normal(size=(4, 4))
    assert abs(float(exact_bracket(P, (0, 1, 2, 3))) - np.linalg.det(P)) < 1e-12
    unit_det = abs(np.linalg.det(P / np.linalg.norm(P, axis=0)))
    assert bracket_below(P, (0, 1, 2, 3), unit_det * 1.000001) is True
    assert bracket_below(P, (0, 1, 2, 3), unit_det * 0.999999) is False   # control: just below must fail


def test_exact_value_is_a_fraction():
    P = np.eye(4) * 0.1
    assert exact_bracket(P, (0, 1, 2, 3)) == Fraction(0.1) ** 4
