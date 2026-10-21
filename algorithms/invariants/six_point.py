"""
algorithms/invariants/six_point.py

Six-point projective invariants J1, J2, J3 for P^3.

For six points p1,...,p6 in general position in P^3, these three bracket-ratio
invariants carry the three degrees of freedom remaining after fixing a projective
frame from four points. Each ratio is constructed so that every coordinate p_i
appears with equal total homogeneous weight in numerator and denominator,
ensuring exact cancellation of det(A) factors under PGL(4,R).

The five-point invariant I5 is NOT defined here. Five points in P^3 have
15 degrees of freedom = dim(PGL(4,R)), leaving no non-trivial projective
invariants. All computation in CAPRI uses six-point configurations.

Reference:
    Quan (1995), Griffiths & Harris (1978), Sturmfels (1993).
"""

from __future__ import annotations
import numpy as np
from algorithms.invariants.brackets import compute_all_brackets_for_tuple, get_bracket
from algorithms.utils import normalize_to_unit_sphere


def evaluate_six_point_invariants(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    denom_tolerance: float = 1e-10
) -> tuple[float, float, float] | None:
    """
    Compute the projective invariant triple (J1, J2, J3) for a six-point tuple.

    J1 = [p1 p2 p3 p4][p1 p2 p5 p6] / ([p1 p2 p3 p5][p1 p2 p4 p6])
    J2 = [p1 p3 p4 p5][p2 p3 p4 p6] / ([p1 p3 p4 p6][p2 p3 p4 p5])
    J3 = [p1 p2 p5 p6][p3 p4 p5 p6] / ([p1 p3 p5 p6][p2 p4 p5 p6])

    Where p1,...,p6 refer to the points at tuple_indices[0],...,tuple_indices[5].

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates.
        tuple_indices: 6-tuple of column indices.
        denom_tolerance: Minimum absolute denominator magnitude. If any
                         denominator is below this, returns None (degenerate).

    Returns:
        (J1, J2, J3) as floats, or None if any denominator is near zero.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6

    points_normalized = normalize_to_unit_sphere(points_homogeneous)
    a, b, c, d, e, f = tuple_indices
    brackets = compute_all_brackets_for_tuple(points_normalized, tuple_indices, pre_normalized=True)

    def B(i: int, j: int, k: int, l: int) -> float:
        return get_bracket(brackets, i, j, k, l)

    # J1 numerator and denominator
    j1_num = B(a, b, c, d) * B(a, b, e, f)
    j1_den = B(a, b, c, e) * B(a, b, d, f)

    # J2 numerator and denominator
    j2_num = B(a, c, d, e) * B(b, c, d, f)
    j2_den = B(a, c, d, f) * B(b, c, d, e)

    # J3 numerator and denominator
    j3_num = B(a, b, e, f) * B(c, d, e, f)
    j3_den = B(a, c, e, f) * B(b, d, e, f)

    # Reject if any denominator is near zero (approaching degeneracy)
    denominators = [j1_den, j2_den, j3_den]
    if any(abs(denom) < denom_tolerance for denom in denominators):
        return None

    j1 = j1_num / j1_den
    j2 = j2_num / j2_den
    j3 = j3_num / j3_den

    return float(j1), float(j2), float(j3)


def evaluate_batch_invariants(
    points_homogeneous: np.ndarray,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
    denom_tolerance: float = 1e-10
) -> list[tuple[float, float, float] | None]:
    """
    Evaluate (J1, J2, J3) for a list of six-point tuples.

    Args:
        points_homogeneous: (4, N) array.
        all_tuple_indices: List of 6-tuples.
        denom_tolerance: Degeneracy rejection threshold.

    Returns:
        List of (J1, J2, J3) or None for each tuple.
    """
    return [
        evaluate_six_point_invariants(points_homogeneous, t, denom_tolerance)
        for t in all_tuple_indices
    ]