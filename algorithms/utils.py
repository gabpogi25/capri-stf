"""
algorithms/utils.py

Shared mathematical utilities for the CAPRI framework.
All functions here are pure mathematical operations with no I/O or side effects.
Any function used in more than one module must live here exclusively.
"""

from __future__ import annotations
import numpy as np


def normalize_to_unit_sphere(points_homogeneous: np.ndarray) -> np.ndarray:
    """
    Normalize each homogeneous column vector to unit Euclidean norm on S^3.

    This is the mandatory pre-processing step before any bracket computation.
    After normalization, every point lies on the unit sphere in R^4, eliminating
    the arbitrary scale factor that otherwise makes bracket magnitudes
    projectively meaningless.

    Mathematically: p_i -> p_i / ||p_i||.  The projective point [p_i] in P^3
    is unchanged, but the representative now satisfies ||p_i|| = 1.

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates in R^4.

    Returns:
        (4, N) array with each column on the unit sphere.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4, (
        f"Expected (4, N) array, got {points_homogeneous.shape}"
    )
    norms = np.linalg.norm(points_homogeneous, axis=0, keepdims=True)
    assert np.all(norms > 1e-15), "Zero-norm homogeneous point encountered."
    return points_homogeneous / norms


def embed_affine_to_homogeneous(points_3d: np.ndarray) -> np.ndarray:
    """
    Embed affine R^3 points into P^3 via the standard chart p = (1, x, y, z).

    Args:
        points_3d: (N, 3) array of affine coordinates.

    Returns:
        (4, N) array of homogeneous coordinates with leading 1.
    """
    assert points_3d.ndim == 2 and points_3d.shape[1] == 3, (
        f"Expected (N, 3) array, got {points_3d.shape}"
    )
    n = points_3d.shape[0]
    ones = np.ones((n, 1))
    homogeneous = np.hstack([ones, points_3d]).T  # (4, N)
    return homogeneous

def whiten_homogeneous_points(points_homogeneous: np.ndarray, max_iterations: int = 500,
                              tolerance: float = 1e-12) -> np.ndarray:
    """
    Canonical projective frame via Tyler's M-estimator of scatter.

    A projective reconstruction (e.g. P1 = [I | 0], P2 from F) is defined only
    up to a 4x4 linear map A, and each point only up to scale. J1, J2, J3 do
    not depend on either, but L(S) depends on A (it is not PGL(4)-invariant),
    so on a raw projective reconstruction L(S) can be arbitrarily large for
    well-spread points.

    Tyler's shape matrix C solves  C = (d / N) sum_i x_i x_i^T / (x_i^T C^{-1} x_i),
    (d = 4, trace normalized). It is scale-free per point and equivariant:
    for Y = A X it returns A C A^T up to a scalar. Hence the whitened points
    C^{-1/2} x_i / ||.|| are the same for every frame up to one orthogonal
    map, under which L(S) is invariant: L(S) becomes a function of the
    projective point set alone. (A plain covariance of unit-normalized points
    is not equivariant, because the normalization depends on the frame.)
    The fixed point exists when no plane (3-dim linear subspace of R^4) holds
    more than 3N/4 of the points, and no line more than N/2.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    unit = normalize_to_unit_sphere(points_homogeneous)
    d, n = unit.shape
    shape = np.eye(d)
    for _ in range(max_iterations):
        inverse = np.linalg.inv(shape)
        weights = np.einsum("in,ij,jn->n", unit, inverse, unit)
        updated = (d / n) * (unit / np.maximum(weights, 1e-300)) @ unit.T
        updated /= np.trace(updated)
        converged = np.max(np.abs(updated - shape)) < tolerance
        shape = updated
        if converged:
            break
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    transform = eigenvectors @ np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, 1e-300))) @ eigenvectors.T
    return normalize_to_unit_sphere(transform @ unit)
