"""
algorithms/invariants/brackets.py

Bracket polynomial computation for projective point configurations in P^3.

A bracket [p1 p2 p3 p4] = det(p1, p2, p3, p4) is the signed volume of
the parallelepiped spanned by four homogeneous representatives in R^4.
Under a projective transformation A, every bracket scales by det(A).
Ratios of brackets with equal total degree are therefore PGL(4,R)-invariant.

All bracket computations in CAPRI operate on unit-sphere-normalized points.
This ensures that bracket magnitudes are geometrically meaningful and
independent of the arbitrary scale of homogeneous representatives.

Reference: Sturmfels (1993), Mumford et al. (1994), Quan (1995).
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from algorithms.backend import get_cpp_module
from algorithms.utils import normalize_to_unit_sphere


def compute_bracket(
    points_homogeneous: np.ndarray,
    indices: tuple[int, int, int, int],
    pre_normalized: bool = False
) -> float:
    """
    Compute a single bracket [p_i p_j p_k p_l] = det of the 4x4 matrix
    formed by stacking the four homogeneous column vectors.

    Points are normalized to the unit sphere before computation unless
    pre_normalized=True, ensuring the bracket value is a geometric volume
    on S^3 independent of coordinate scaling.

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates.
        indices: 4-tuple of column indices selecting which points to use.
        pre_normalized: If True, skip normalization (caller guarantees
                        points are already on the unit sphere).

    Returns:
        Scalar bracket value (float).
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4, (
        f"Expected (4, N), got {points_homogeneous.shape}"
    )
    assert len(indices) == 4, "Bracket requires exactly 4 point indices."

    if not pre_normalized:
        points_homogeneous = normalize_to_unit_sphere(points_homogeneous)

    i, j, k, l = indices
    matrix = points_homogeneous[:, [i, j, k, l]]  # (4, 4)
    return float(np.linalg.det(matrix))


def compute_all_brackets_for_tuple(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    pre_normalized: bool = False
) -> dict[tuple[int, int, int, int], float]:
    """
    Compute all C(6,4) = 15 brackets for a six-point tuple.
    This is the complete bracket table needed for J1, J2, J3 evaluation
    and for computing delta_min(S).

    Points are normalized once, then all 15 brackets are computed on the
    normalized representatives.

    Args:
        points_homogeneous: (4, N) array.
        tuple_indices: 6-tuple of column indices for the six points.
        pre_normalized: If True, skip normalization.

    Returns:
        Dictionary mapping each 4-index combination to its bracket value.
    """
    assert len(tuple_indices) == 6, "Six-point tuple requires exactly 6 indices."

    capri_cpp = get_cpp_module()
    if capri_cpp is not None:
        # The C++ routine enumerates quads in the order of the indices it is
        # given, while the table is keyed by sorted quads. Sort first so each
        # value lands under the right key (unsorted input would produce
        # mislabelled brackets on the C++ path only).
        sorted_tuple = tuple(sorted(tuple_indices))
        cpp_result = capri_cpp.compute_all_brackets(points_homogeneous, sorted_tuple)
        return {quad: val for quad, val in zip(combinations(sorted_tuple, 4), cpp_result)}

    if not pre_normalized:
        points_homogeneous = normalize_to_unit_sphere(points_homogeneous)

    tuple_indices = tuple(sorted(tuple_indices))
    brackets: dict[tuple[int, int, int, int], float] = {}
    for quad in combinations(tuple_indices, 4):
        brackets[quad] = compute_bracket(points_homogeneous, quad, pre_normalized=True)
    return brackets


def get_bracket(
    bracket_table: dict[tuple[int, int, int, int], float],
    i: int, j: int, k: int, l: int
) -> float:
    """
    Retrieve a bracket from a precomputed bracket table.
    Handles index ordering: the table stores indices in sorted order.

    Args:
        bracket_table: Output of compute_all_brackets_for_tuple.
        i, j, k, l: Point indices (any order).

    Returns:
        Bracket value, with sign adjusted for permutation parity.
    """
    key_sorted = tuple(sorted([i, j, k, l]))

    # Compute parity of permutation from [i,j,k,l] to sorted order
    perm = [i, j, k, l]
    sorted_perm = sorted(range(4), key=lambda x: perm[x])
    # Count inversions for sign
    inversions = 0
    for a in range(4):
        for b in range(a + 1, 4):
            if sorted_perm[a] > sorted_perm[b]:
                inversions += 1
    sign = (-1) ** inversions

    assert key_sorted in bracket_table, (
        f"Bracket {key_sorted} not found in table. "
        f"Available: {list(bracket_table.keys())[:5]}..."
    )
    return sign * bracket_table[key_sorted]