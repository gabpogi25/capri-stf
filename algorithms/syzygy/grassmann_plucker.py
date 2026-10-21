"""
algorithms/syzygy/grassmann_plucker.py

Grassmann-Plucker relations for six-point bracket tables in P^3.

The 15 four-point brackets [p_i p_j p_k p_l] arising from 6 points in P^3
are NOT algebraically independent. They are the Plucker coordinates of a
point on the Grassmannian Gr(4,6), and must satisfy the Grassmann-Plucker
quadratic relations.

For any pair of fixed indices (a, b) and four free indices (c, d, e, f)
chosen from the six points, the following 3-term identity holds:

    [abcd][abef] - [abce][abdf] + [abcf][abde] = 0

This is the Gr(2,4) Plucker relation applied to the quotient space
R^4 / span(p_a, p_b). Via the Hodge star isomorphism Gr(4,6) ~ Gr(2,6),
these are equivalent to the standard Plucker relations for 2-forms.

There are C(6,2) = 15 such relations (one per fixed pair). The
Grassmannian Gr(4,6) has dimension 8 in P^14, so at least 6 of these
are algebraically independent (codimension = 14 - 8 = 6). The
implementation evaluates all 15 for maximum diagnostic sensitivity.

The Plucker residual serves as a diagnostic for:
  - Internal bracket consistency (are the brackets geometrically valid?)
  - Numerical error detection (corruption in bracket computation)
  - Mismatch detection (brackets from incompatible configurations)

This is COMPLEMENTARY to the multi-view consistency residual rho, which
checks agreement of the invariants J1, J2, J3 across independent views.
The Plucker residual checks the algebraic validity of the raw brackets
from a SINGLE measurement.

Reference:
    Griffiths & Harris (1978), Sturmfels (1993), Hodge & Pedoe (1952).
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from algorithms.invariants.brackets import compute_all_brackets_for_tuple, get_bracket
from algorithms.utils import normalize_to_unit_sphere


def compute_single_plucker_relation(
    bracket_table: dict[tuple[int, int, int, int], float],
    fixed_a: int,
    fixed_b: int,
    free_c: int,
    free_d: int,
    free_e: int,
    free_f: int,
) -> float:
    """
    Evaluate one Grassmann-Plucker relation for the given index assignment.

    The identity is:
        [abcd][abef] - [abce][abdf] + [abcf][abde] = 0

    Args:
        bracket_table: Precomputed bracket table from compute_all_brackets_for_tuple.
        fixed_a, fixed_b: The two fixed indices.
        free_c, free_d, free_e, free_f: The four free indices.

    Returns:
        The residual value (should be near zero for valid configurations).
    """
    def B(i: int, j: int, k: int, l: int) -> float:
        return get_bracket(bracket_table, i, j, k, l)

    a, b, c, d, e, f = fixed_a, fixed_b, free_c, free_d, free_e, free_f

    term_1 = B(a, b, c, d) * B(a, b, e, f)
    term_2 = B(a, b, c, e) * B(a, b, d, f)
    term_3 = B(a, b, c, f) * B(a, b, d, e)

    return float(term_1 - term_2 + term_3)


def compute_all_plucker_relations(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    pre_normalized: bool = False,
) -> list[dict]:
    """
    Evaluate all 15 Grassmann-Plucker relations for a six-point tuple.

    For each of the C(6,2) = 15 choices of fixed pair (a, b), evaluates
    the 3-term relation on the remaining four free indices.

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates.
        tuple_indices: 6-tuple of column indices for the six points.
        pre_normalized: If True, skip unit-sphere normalization.

    Returns:
        List of 15 dicts, each containing:
          - 'fixed_pair': (a, b) indices
          - 'free_quad': (c, d, e, f) indices
          - 'residual': float (should be near zero)
          - 'terms': (term_1, term_2, term_3) for diagnostics
    """
    assert len(tuple_indices) == 6

    if not pre_normalized:
        points_homogeneous = normalize_to_unit_sphere(points_homogeneous)

    brackets = compute_all_brackets_for_tuple(
        points_homogeneous, tuple_indices, pre_normalized=True
    )

    idx_list = list(tuple_indices)
    results: list[dict] = []

    for pair in combinations(range(6), 2):
        local_a, local_b = pair
        remaining = [i for i in range(6) if i not in pair]
        local_c, local_d, local_e, local_f = remaining

        a = idx_list[local_a]
        b = idx_list[local_b]
        c = idx_list[local_c]
        d = idx_list[local_d]
        e = idx_list[local_e]
        f = idx_list[local_f]

        def B(i: int, j: int, k: int, l: int) -> float:
            return get_bracket(brackets, i, j, k, l)

        term_1 = B(a, b, c, d) * B(a, b, e, f)
        term_2 = B(a, b, c, e) * B(a, b, d, f)
        term_3 = B(a, b, c, f) * B(a, b, d, e)
        residual = term_1 - term_2 + term_3

        results.append({
            "fixed_pair": (a, b),
            "free_quad": (c, d, e, f),
            "residual": float(residual),
            "terms": (float(term_1), float(term_2), float(term_3)),
        })

    return results


def compute_plucker_residual(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
) -> float:
    """
    Compute the maximum absolute Plucker residual across all 15 relations.

    This is the primary scalar diagnostic for bracket-level algebraic
    consistency. For an exact configuration in exact arithmetic, this is
    identically zero. Under floating-point arithmetic, values below 1e-10
    indicate healthy bracket computation. Values above 1e-6 indicate
    numerical problems or data corruption.

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates.
        tuple_indices: 6-tuple of column indices.

    Returns:
        max |residual| over all 15 Grassmann-Plucker relations (float >= 0).
    """
    relations = compute_all_plucker_relations(points_homogeneous, tuple_indices)
    return float(max(abs(r["residual"]) for r in relations))


def batch_plucker_residuals(
    points_homogeneous: np.ndarray,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
) -> np.ndarray:
    """
    Compute the max Plucker residual for each tuple in a list.

    Args:
        points_homogeneous: (4, N) array.
        all_tuple_indices: List of 6-tuples.

    Returns:
        (K,) array of max Plucker residuals, one per tuple.
    """
    return np.array([
        compute_plucker_residual(points_homogeneous, t)
        for t in all_tuple_indices
    ])
