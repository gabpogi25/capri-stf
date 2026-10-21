"""
algorithms/invariants/six_point_torch.py

Differentiable six-point projective invariant evaluation in PyTorch.

Computes the invariant triple (J1, J2, J3) with full gradient support,
enabling end-to-end training of downstream classifiers (e.g., the
ConfidenceFusionClassifier) through the algebraic invariant computation.

The invariant formulas are identical to six_point.py:

    J1 = [p1 p2 p3 p4][p1 p2 p5 p6] / ([p1 p2 p3 p5][p1 p2 p4 p6])
    J2 = [p1 p3 p4 p5][p2 p3 p4 p6] / ([p1 p3 p4 p6][p2 p3 p4 p5])
    J3 = [p1 p2 p5 p6][p3 p4 p5 p6] / ([p1 p3 p5 p6][p2 p4 p5 p6])

Gradients flow through torch.linalg.det (gradient = cofactor matrix)
and through the ratio operations. Denominators near zero return None,
same as the numpy version.

Reference: Quan (1995), Griffiths & Harris (1978), Sturmfels (1993).
"""

from __future__ import annotations
import torch
from itertools import combinations
from algorithms.utils_torch import normalize_to_unit_sphere_torch


# Precomputed C(6,4) = 15 local quad indices for bracket evaluation
_QUAD_COMBOS: list[tuple[int, int, int, int]] = list(combinations(range(6), 4))


def _compute_bracket_torch(
    points_normalized: torch.Tensor,
    i: int, j: int, k: int, l: int,
) -> torch.Tensor:
    """
    Compute a single bracket [pi pj pk pl] as a differentiable scalar tensor.

    Args:
        points_normalized: (4, N) unit-sphere-normalized tensor.
        i, j, k, l: Column indices.

    Returns:
        Scalar tensor (0-dim) with gradient support.
    """
    matrix = points_normalized[:, [i, j, k, l]]  # (4, 4)
    return torch.linalg.det(matrix)


def evaluate_six_point_invariants_torch(
    points_homogeneous: torch.Tensor,
    tuple_indices: tuple[int, int, int, int, int, int],
    denom_tolerance: float = 1e-10,
) -> torch.Tensor | None:
    """
    Compute (J1, J2, J3) as a differentiable (3,) tensor.

    Args:
        points_homogeneous: (4, N) tensor. Will be unit-sphere normalized.
        tuple_indices: 6-tuple of column indices.
        denom_tolerance: Minimum absolute denominator. Returns None if any
                         denominator is below this threshold.

    Returns:
        (3,) tensor of [J1, J2, J3] with gradients, or None if degenerate.
    """
    assert points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6

    points_normalized = normalize_to_unit_sphere_torch(points_homogeneous)
    a, b, c, d, e, f = tuple_indices

    def B(i: int, j: int, k: int, l: int) -> torch.Tensor:
        return _compute_bracket_torch(points_normalized, i, j, k, l)

    # J1 = [abcd][abef] / ([abce][abdf])
    j1_num = B(a, b, c, d) * B(a, b, e, f)
    j1_den = B(a, b, c, e) * B(a, b, d, f)

    # J2 = [acde][bcdf] / ([acdf][bcde])
    j2_num = B(a, c, d, e) * B(b, c, d, f)
    j2_den = B(a, c, d, f) * B(b, c, d, e)

    # J3 = [abef][cdef] / ([acef][bdef])
    j3_num = B(a, b, e, f) * B(c, d, e, f)
    j3_den = B(a, c, e, f) * B(b, d, e, f)

    # Check denominators
    denoms = [j1_den, j2_den, j3_den]
    if any(torch.abs(d) < denom_tolerance for d in denoms):
        return None

    j1 = j1_num / j1_den
    j2 = j2_num / j2_den
    j3 = j3_num / j3_den

    return torch.stack([j1, j2, j3])


def evaluate_batch_invariants_torch(
    points_homogeneous: torch.Tensor,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
    denom_tolerance: float = 1e-10,
) -> tuple[torch.Tensor, list[bool]]:
    """
    Evaluate (J1, J2, J3) for a batch of tuples.

    Returns a (K, 3) tensor where degenerate tuples are filled with zeros,
    and a mask indicating which tuples are valid.

    Args:
        points_homogeneous: (4, N) tensor.
        all_tuple_indices: List of K 6-tuples.
        denom_tolerance: Degeneracy rejection threshold.

    Returns:
        (invariants, valid_mask):
          - invariants: (K, 3) tensor.
          - valid_mask: List of K booleans (True = non-degenerate).
    """
    n_tuples = len(all_tuple_indices)
    invariants = torch.zeros(
        (n_tuples, 3),
        dtype=points_homogeneous.dtype,
        device=points_homogeneous.device,
    )
    valid_mask: list[bool] = []

    for k, t in enumerate(all_tuple_indices):
        result = evaluate_six_point_invariants_torch(
            points_homogeneous, t, denom_tolerance
        )
        if result is not None:
            invariants[k] = result
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    return invariants, valid_mask
