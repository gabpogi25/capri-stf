"""
algorithms/utils_torch.py

PyTorch-differentiable versions of shared mathematical utilities.

These are direct translations of the functions in algorithms/utils.py,
reimplemented using torch operations so that gradients flow through
to support end-to-end training of the confidence fusion classifier
with invariant-aware losses.

All functions here mirror their numpy counterparts exactly in behavior.
"""

from __future__ import annotations
import torch


def normalize_to_unit_sphere_torch(points_homogeneous: torch.Tensor) -> torch.Tensor:
    """
    Normalize each homogeneous column vector to unit Euclidean norm on S^3.

    Differentiable: gradients flow through the normalization.

    Args:
        points_homogeneous: (4, N) tensor of homogeneous coordinates.

    Returns:
        (4, N) tensor with each column on the unit sphere in R^4.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4, (
        f"Expected (4, N) tensor, got {points_homogeneous.shape}"
    )
    norms = torch.linalg.norm(points_homogeneous, dim=0, keepdim=True)
    assert torch.all(norms > 1e-15), "Zero-norm homogeneous point encountered."
    return points_homogeneous / norms


def embed_affine_to_homogeneous_torch(points_3d: torch.Tensor) -> torch.Tensor:
    """
    Embed affine R^3 points into P^3 via the standard chart p = (1, x, y, z).

    Differentiable: gradients flow through the embedding.

    Args:
        points_3d: (N, 3) tensor of affine coordinates.

    Returns:
        (4, N) tensor of homogeneous coordinates with leading 1.
    """
    assert points_3d.ndim == 2 and points_3d.shape[1] == 3, (
        f"Expected (N, 3) tensor, got {points_3d.shape}"
    )
    n = points_3d.shape[0]
    ones = torch.ones((n, 1), dtype=points_3d.dtype, device=points_3d.device)
    homogeneous = torch.cat([ones, points_3d], dim=1).T  # (4, N)
    return homogeneous
