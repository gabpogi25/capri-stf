"""
algorithms/degeneracy/stratification.py

Degeneracy stratum classification for six-point projective configurations.

Configurations in (P^3)^6 are organized into strata by their bracket-vanishing
pattern. A bracket [pi pj pk pl] vanishes when the four homogeneous
representatives become linearly dependent in R^4. Different patterns of
vanishing brackets correspond to different geometric degeneracy types, organized
into a filtration of the vanishing locus V(B_6).

Strata (docs/MATHEMATICS.md):
    GENERIC:   All 15 four-point brackets nonzero. Stable computation.
    BOUNDARY:  At least one bracket near zero. Warning: instability approaching.
    COMPOUND:  Multiple dependent quadruples. Compounding instability.
    CRITICAL:  Denominator brackets of J1/J2/J3 near zero. Invariants undefined.

Reference: Hartshorne (1977), Oxley (2011); docs/MATHEMATICS.md.
"""

from __future__ import annotations
import numpy as np
from enum import Enum
from itertools import combinations
from algorithms.invariants.brackets import compute_bracket
from algorithms.backend import get_cpp_module
from algorithms.utils import normalize_to_unit_sphere


class DegeneracyStratum(Enum):
    """Enumeration of degeneracy strata ordered by increasing instability."""
    GENERIC = "generic"
    BOUNDARY = "boundary"
    COMPOUND = "compound"
    CRITICAL = "critical"


# Denominator bracket index-sets for J1, J2, J3.
# These are the brackets that appear in the denominators of the three invariants.
# If any of these is near zero, the invariants are undefined (CRITICAL stratum).
_DENOMINATOR_BRACKET_PATTERNS = [
    # J1 denominators
    (0, 1, 2, 4),  # [p1 p2 p3 p5]
    (0, 1, 3, 5),  # [p1 p2 p4 p6]
    # J2 denominators
    (0, 2, 3, 5),  # [p1 p3 p4 p6]
    (1, 2, 3, 4),  # [p2 p3 p4 p5]
    # J3 denominators
    (0, 2, 4, 5),  # [p1 p3 p5 p6]
    (1, 3, 4, 5),  # [p2 p4 p5 p6]
]


def classify_degeneracy_stratum(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    near_zero_threshold: float = 1e-6,
    boundary_threshold: float = 1e-2
) -> DegeneracyStratum:
    """
    Classify the degeneracy stratum of a six-point tuple by bracket analysis.

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates.
        tuple_indices: 6-tuple of column indices.
        near_zero_threshold: Bracket magnitude below which the bracket is
                             treated as vanished (CRITICAL boundary).
        boundary_threshold: Bracket magnitude below which the bracket is
                            treated as near-zero (BOUNDARY/COMPOUND boundary).

    Returns:
        DegeneracyStratum enum value.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6

    capri_cpp = get_cpp_module()
    if capri_cpp is not None:
        cpp_stratum = capri_cpp.classify_degeneracy_stratum(
            points_homogeneous, tuple_indices, near_zero_threshold, boundary_threshold
        )
        # Convert C++ enum to Python enum by matching the name
        return DegeneracyStratum[cpp_stratum.name]

    points_normalized = normalize_to_unit_sphere(points_homogeneous)
    idx = tuple_indices  # local alias for readability

    # Check denominator brackets first (CRITICAL stratum)
    for local_quad in _DENOMINATOR_BRACKET_PATTERNS:
        global_quad = tuple(idx[i] for i in local_quad)
        val = abs(compute_bracket(points_normalized, global_quad, pre_normalized=True))
        if val < near_zero_threshold:
            return DegeneracyStratum.CRITICAL

    # Count all near-zero brackets
    near_zero_count = 0
    for quad in combinations(tuple_indices, 4):
        val = abs(compute_bracket(points_normalized, quad, pre_normalized=True))
        if val < boundary_threshold:
            near_zero_count += 1

    if near_zero_count == 0:
        return DegeneracyStratum.GENERIC
    elif near_zero_count == 1:
        return DegeneracyStratum.BOUNDARY
    else:
        return DegeneracyStratum.COMPOUND


def batch_classify_strata(
    points_homogeneous: np.ndarray,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
    near_zero_threshold: float = 1e-6,
    boundary_threshold: float = 1e-2
) -> list[DegeneracyStratum]:
    """
    Classify strata for a list of six-point tuples.

    Args:
        points_homogeneous: (4, N) array.
        all_tuple_indices: List of 6-tuples.

    Returns:
        List of DegeneracyStratum values, one per tuple.
    """
    capri_cpp = get_cpp_module()
    if capri_cpp is not None:
        cpp_strata = capri_cpp.batch_classify_strata(
            points_homogeneous, all_tuple_indices, near_zero_threshold, boundary_threshold
        )
        return [DegeneracyStratum[s.name] for s in cpp_strata]

    return [
        classify_degeneracy_stratum(
            points_homogeneous, t, near_zero_threshold, boundary_threshold
        )
        for t in all_tuple_indices
    ]


def filter_by_stratum(
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
    strata: list[DegeneracyStratum],
    allowed_strata: set[DegeneracyStratum] = None
) -> list[tuple[int, int, int, int, int, int]]:
    """
    Return only those tuples whose stratum is in allowed_strata.

    Args:
        all_tuple_indices: List of 6-tuples.
        strata: Corresponding list of DegeneracyStratum values.
        allowed_strata: Set of acceptable strata. Defaults to {GENERIC, BOUNDARY}.

    Returns:
        Filtered list of 6-tuples.
    """
    if allowed_strata is None:
        allowed_strata = {DegeneracyStratum.GENERIC, DegeneracyStratum.BOUNDARY}
    return [
        t for t, s in zip(all_tuple_indices, strata)
        if s in allowed_strata
    ]

def largest_coplanar_subset_size(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    tolerance: float,
) -> int:
    """
    m_hat = max{|Q| : Q subset of the tuple, |Q| >= 4, sigma_4(P_Q) <= tolerance},
    or 0 if no 4-subset is that close to a plane.

    P_Q is the 4 x |Q| matrix of unit-normalized representatives. By
    Eckart-Young, sigma_4(P_Q) is the Frobenius distance of Q to the nearest
    all-coplanar configuration, so m_hat is the size of the largest subset of
    the tuple that is coplanar up to `tolerance`.

    Under bounded noise: with every column perturbed by at most sigma, a truly
    coplanar Q satisfies sigma_4 <= sqrt(|Q|) * sigma, so tolerance =
    sqrt(6) * sigma gives no false negatives under bounded noise.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6
    normalized = normalize_to_unit_sphere(points_homogeneous[:, list(tuple_indices)])
    for size in (6, 5, 4):
        for subset in combinations(range(6), size):
            if np.linalg.svd(normalized[:, list(subset)], compute_uv=False)[3] <= tolerance:
                return size
    return 0


def classify_stratum_by_rank(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    tolerance: float = 1e-6,
) -> DegeneracyStratum:
    """
    Rank-based stratum: m_hat = 6 -> CRITICAL, 5 -> COMPOUND, 4 -> BOUNDARY,
    otherwise GENERIC. Unlike the single global bracket threshold of
    classify_degeneracy_stratum, which classifies the compound stratum with
    0% recall, it separates the strata by subset size.
    """
    m_hat = largest_coplanar_subset_size(points_homogeneous, tuple_indices, tolerance)
    return {6: DegeneracyStratum.CRITICAL, 5: DegeneracyStratum.COMPOUND,
            4: DegeneracyStratum.BOUNDARY}.get(m_hat, DegeneracyStratum.GENERIC)
