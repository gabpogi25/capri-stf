"""
algorithms/invariants/brackets_torch.py

GPU-accelerated batched bracket computation using PyTorch.

Replaces the sequential Python loop over numpy.linalg.det calls with a
single batched torch.linalg.det call on a (K*15, 4, 4) tensor. This
reduces K*15 sequential LAPACK calls to a single batched call, achieving
up to 100x speedup for K >= 1000 tuples (GOAL.md Section 9).

All operations use torch.Tensor and remain CPU-compatible. GPU acceleration
is opt-in via .to(device). Gradients flow through torch.linalg.det for
use in the differentiable pipeline (Track 4).

Reference: GOAL.md Section 9 (Bracket Computation Vectorization).
"""

from __future__ import annotations
import torch
import numpy as np
from itertools import combinations


# Precompute the 15 local index combinations for a 6-point tuple.
# These are the C(6,4) = 15 ways to choose 4 indices from {0,1,2,3,4,5}.
_QUAD_COMBOS: list[tuple[int, int, int, int]] = list(combinations(range(6), 4))


def _normalize_to_unit_sphere_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize each column of a (4, N) tensor to unit Euclidean norm.

    Args:
        points: (4, N) tensor of homogeneous coordinates.

    Returns:
        (4, N) tensor with each column on the unit sphere in R^4.
    """
    norms = torch.linalg.norm(points, dim=0, keepdim=True)
    assert torch.all(norms > 1e-15), "Zero-norm homogeneous point encountered."
    return points / norms


def compute_all_brackets_batched(
    points_homogeneous: torch.Tensor,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
) -> torch.Tensor:
    """
    Compute all 15 brackets for each of K six-point tuples in one batch.

    Builds a (K*15, 4, 4) tensor of bracket matrices and calls
    torch.linalg.det once. Returns a (K, 15) tensor of bracket values.

    The 15 brackets per tuple correspond to the C(6,4) four-point
    sub-selections, enumerated in lexicographic order of the local
    indices {0,1,2,3,4,5} within each tuple.

    Args:
        points_homogeneous: (4, N) tensor of homogeneous coordinates.
            Will be normalized to the unit sphere before computation.
        all_tuple_indices: List of K 6-tuples, each containing column
            indices into points_homogeneous.

    Returns:
        (K, 15) tensor of bracket values (determinants). Gradients flow
        through to the input points if requires_grad=True.
    """
    assert points_homogeneous.shape[0] == 4

    points_normalized = _normalize_to_unit_sphere_torch(points_homogeneous)

    n_tuples = len(all_tuple_indices)
    # Build all bracket matrices: (K*15, 4, 4)
    matrices = torch.empty(
        (n_tuples * 15, 4, 4),
        dtype=points_normalized.dtype,
        device=points_normalized.device,
    )

    for k, t in enumerate(all_tuple_indices):
        for q_idx, quad_local in enumerate(_QUAD_COMBOS):
            global_indices = [t[i] for i in quad_local]
            matrices[k * 15 + q_idx] = points_normalized[:, global_indices].T

    # Single batched determinant call
    all_dets = torch.linalg.det(matrices)  # (K*15,)

    return all_dets.reshape(n_tuples, 15)


def compute_conditioning_scores_batched(
    points_homogeneous: torch.Tensor,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
) -> torch.Tensor:
    """
    Compute conditioning scores L(S) for K tuples via batched brackets.

    L(S) = 8.0 / delta_min(S)^2, where delta_min is the minimum absolute
    bracket magnitude across the 15 four-point sub-selections.

    Args:
        points_homogeneous: (4, N) tensor.
        all_tuple_indices: List of K 6-tuples.

    Returns:
        (K,) tensor of conditioning scores. Values are inf where delta_min
        is effectively zero.
    """
    brackets = compute_all_brackets_batched(points_homogeneous, all_tuple_indices)
    delta_min = torch.min(torch.abs(brackets), dim=1).values  # (K,)

    # Avoid division by zero
    safe_delta_min = torch.clamp(delta_min, min=1e-15)
    scores = 8.0 / (safe_delta_min ** 2)

    # Set inf where delta_min was effectively zero
    scores = torch.where(delta_min < 1e-15, torch.tensor(float('inf')), scores)

    return scores


def classify_strata_batched(
    points_homogeneous: torch.Tensor,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]],
    near_zero_threshold: float = 1e-6,
    boundary_threshold: float = 1e-2,
) -> list[str]:
    """
    Classify degeneracy strata for K tuples using batched bracket computation.

    Uses the same classification logic as stratification.py but computed
    from the batched bracket tensor for efficiency.

    The denominator bracket patterns (local indices within a 6-tuple) for
    J1, J2, J3 are checked first for CRITICAL stratum detection.

    Args:
        points_homogeneous: (4, N) tensor.
        all_tuple_indices: List of K 6-tuples.
        near_zero_threshold: CRITICAL stratum boundary.
        boundary_threshold: BOUNDARY/COMPOUND stratum boundary.

    Returns:
        List of K stratum strings: "generic", "boundary", "compound", "critical".
    """
    brackets = compute_all_brackets_batched(points_homogeneous, all_tuple_indices)
    abs_brackets = torch.abs(brackets)  # (K, 15)

    # Denominator bracket local indices (matching stratification.py)
    # _QUAD_COMBOS maps local quad -> position in the 15-element list.
    # We need to find which positions correspond to denominator brackets.
    denom_local_quads = [
        (0, 1, 2, 4),  # [p1 p2 p3 p5]
        (0, 1, 3, 5),  # [p1 p2 p4 p6]
        (0, 2, 3, 5),  # [p1 p3 p4 p6]
        (1, 2, 3, 4),  # [p2 p3 p4 p5]
        (0, 2, 4, 5),  # [p1 p3 p5 p6]
        (1, 3, 4, 5),  # [p2 p4 p5 p6]
    ]
    denom_positions = [_QUAD_COMBOS.index(q) for q in denom_local_quads]

    strata: list[str] = []
    for k in range(len(all_tuple_indices)):
        # Check CRITICAL: any denominator bracket near zero
        is_critical = any(
            abs_brackets[k, pos].item() < near_zero_threshold
            for pos in denom_positions
        )
        if is_critical:
            strata.append("critical")
            continue

        # Count near-zero brackets
        near_zero_count = int((abs_brackets[k] < boundary_threshold).sum().item())
        if near_zero_count == 0:
            strata.append("generic")
        elif near_zero_count == 1:
            strata.append("boundary")
        else:
            strata.append("compound")

    return strata
