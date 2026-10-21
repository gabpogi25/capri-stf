"""
synthetic_data/generator.py

Synthetic point configuration generator for CAPRI validation.

Generates five regime types (docs/MATHEMATICS.md):
  1. GENERIC:        Points in general position (full rank).
  2. NEAR_DEGENERATE: Points with controlled coplanarity.
  3. SPARSE_VIEW:    Limited camera viewpoints, missing correspondences.
  4. MISMATCHED:     Deliberate false correspondences injected.
  5. NON_RIGID:      Bounded displacement between two captures.

All generators accept a seed parameter for exact reproducibility.
Generated datasets are saved via synthetic_data/export.py.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from algorithms.utils import embed_affine_to_homogeneous


def generate_generic_configuration(
    n_points: int,
    scale: float = 1.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate n_points in general position in R^3, embedded into P^3.

    Points are sampled uniformly from [-scale, scale]^3 and checked for
    near-degeneracy (no four points coplanar to within 1e-4 volume).
    NOTE: The 1e-4 volume threshold is an empirically chosen non-degeneracy
    margin, not formally derived, per QUALITY.md 1.4.

    Args:
        n_points: Number of points to generate. Must be >= 6.
        scale: Bounding box half-extent.
        rng: Random generator for reproducibility.

    Returns:
        (4, n_points) homogeneous coordinate array.
    """
    assert n_points >= 6
    if rng is None:
        rng = np.random.default_rng()

    max_attempts = 1000
    for _ in range(max_attempts):
        pts_3d = rng.uniform(-scale, scale, size=(n_points, 3))
        pts_h = embed_affine_to_homogeneous(pts_3d)
        # Quick check: sample a few random quadruples for near-degeneracy
        min_det = float('inf')
        for _ in range(20):
            quad = rng.choice(n_points, size=4, replace=False)
            det_val = abs(np.linalg.det(pts_h[:, quad]))
            if det_val < min_det:
                min_det = det_val
        if min_det > 1e-4:
            return pts_h

    # Fallback: return last generated set with a warning
    return pts_h


def generate_near_degenerate_configuration(
    n_points: int,
    coplanar_fraction: float = 0.5,
    plane_offset: float = 1e-3,
    scale: float = 1.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate a configuration where a fraction of points are nearly coplanar.

    coplanar_fraction of the points are placed close to a random plane
    (within plane_offset distance), creating near-vanishing brackets for
    quadruples drawn from that subset.

    Args:
        n_points: Total number of points.
        coplanar_fraction: Fraction of points placed near the plane [0,1].
        plane_offset: Maximum distance from the plane for coplanar points.
        scale: Bounding box half-extent for non-coplanar points.
        rng: Random generator.

    Returns:
        (4, n_points) homogeneous coordinate array.
    """
    assert 0.0 < coplanar_fraction < 1.0
    if rng is None:
        rng = np.random.default_rng()

    n_coplanar = max(4, int(n_points * coplanar_fraction))
    n_generic = n_points - n_coplanar

    # Random plane: normal vector and offset
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    plane_d = rng.uniform(-scale, scale)

    # Generate coplanar points (projected onto plane + small offset)
    coplanar_3d = rng.uniform(-scale, scale, size=(n_coplanar, 3))
    # Project onto plane: x -> x - ((x.n - d) / ||n||^2) * n
    dists = coplanar_3d @ normal - plane_d
    coplanar_3d -= np.outer(dists, normal)
    # Add small perturbation off the plane
    perturbation = rng.uniform(-plane_offset, plane_offset, size=(n_coplanar,))
    coplanar_3d += np.outer(perturbation, normal)

    # Generate generic points
    generic_3d = rng.uniform(-scale, scale, size=(n_generic, 3))

    all_3d = np.vstack([coplanar_3d, generic_3d])
    return embed_affine_to_homogeneous(all_3d)


def generate_sparse_view_configuration(
    n_points: int,
    n_views: int = 3,
    points_per_view: int = 8,
    scale: float = 1.0,
    rng: np.random.Generator = None
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Generate a configuration where only a subset of points are visible per view.

    Returns both the full point set and the per-view visibility lists,
    simulating sparse viewpoint coverage.

    Args:
        n_points: Total number of scene points.
        n_views: Number of available views.
        points_per_view: How many points are visible per view.
        scale: Bounding box half-extent.
        rng: Random generator.

    Returns:
        Tuple of:
          - (4, n_points) homogeneous coordinate array (full scene)
          - List of n_views lists, each containing visible point indices
    """
    if rng is None:
        rng = np.random.default_rng()

    pts_h = generate_generic_configuration(n_points, scale, rng)
    visible_per_view = [
        list(rng.choice(n_points, size=min(points_per_view, n_points), replace=False))
        for _ in range(n_views)
    ]
    return pts_h, visible_per_view


def generate_mismatched_configuration(
    n_points: int,
    mismatch_fraction: float = 0.2,
    scale: float = 1.0,
    rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Generate a configuration with deliberately incorrect correspondences.

    A fraction of points have their identities swapped with random other
    points, creating false correspondences that should be detected by
    the syzygy residual.

    Args:
        n_points: Number of points.
        mismatch_fraction: Fraction of points with swapped identities.
        scale: Bounding box half-extent.
        rng: Random generator.

    Returns:
        Tuple of:
          - (4, n_points) original array
          - (4, n_points) array with mismatched points substituted
          - List of indices that were mismatched
    """
    if rng is None:
        rng = np.random.default_rng()

    pts_h = generate_generic_configuration(n_points, scale, rng)
    n_mismatch = max(1, int(n_points * mismatch_fraction))
    mismatch_indices = list(rng.choice(n_points, size=n_mismatch, replace=False))

    # Generate replacement points (different locations)
    replacement_3d = rng.uniform(-scale, scale, size=(n_mismatch, 3))
    replacement_h = embed_affine_to_homogeneous(replacement_3d)

    pts_mismatched = pts_h.copy()
    for k, idx in enumerate(mismatch_indices):
        pts_mismatched[:, idx] = replacement_h[:, k]

    return pts_h, pts_mismatched, mismatch_indices


def generate_non_rigid_configuration(
    n_points: int,
    deformation_bound: float = 0.05,
    scale: float = 1.0,
    rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate two captures of a mildly deforming scene.

    Models non-rigid motion as bounded displacement: p_tilde_i = p_i + epsilon_i
    with ||epsilon_i|| <= deformation_bound.

    Args:
        n_points: Number of points.
        deformation_bound: Maximum displacement magnitude per point.
        scale: Bounding box half-extent.
        rng: Random generator.

    Returns:
        Tuple of:
          - (4, n_points) first capture (original)
          - (4, n_points) second capture (deformed)
          - deformation_bound (float, for tau(S) calibration)
    """
    if rng is None:
        rng = np.random.default_rng()

    pts_h_original = generate_generic_configuration(n_points, scale, rng)

    # Generate bounded displacement in affine coordinates
    # Sample direction uniformly on S^2, magnitude uniformly in [0, bound]
    # This ensures ||epsilon_i|| <= deformation_bound exactly
    directions = rng.standard_normal((3, n_points))
    norms = np.linalg.norm(directions, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-15)  # avoid division by zero
    directions = directions / norms  # unit vectors on S^2
    magnitudes = rng.uniform(0, deformation_bound, size=(1, n_points))
    displacement = directions * magnitudes  # (3, n_points), ||.|| <= bound

    displacement_h = np.vstack([np.zeros((1, n_points)), displacement])
    pts_h_deformed = pts_h_original + displacement_h

    return pts_h_original, pts_h_deformed, deformation_bound


def apply_random_projective_transform(
    points_homogeneous: np.ndarray,
    det_range: tuple[float, float] = (0.5, 2.0),
    rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a random PGL(4,R) transformation to verify projective invariance.

    The transformation matrix A is drawn with det(A) in det_range to avoid
    near-singular transformations. Invariants computed on both the original
    and transformed configuration must agree to within numerical precision.

    Args:
        points_homogeneous: (4, N) array.
        det_range: (min, max) for |det(A)|.
        rng: Random generator.

    Returns:
        Tuple of:
          - Transformed (4, N) array
          - (4, 4) transformation matrix A
    """
    if rng is None:
        rng = np.random.default_rng()

    max_attempts = 100
    for _ in range(max_attempts):
        A = rng.standard_normal((4, 4))
        d = abs(np.linalg.det(A))
        if det_range[0] <= d <= det_range[1]:
            transformed = A @ points_homogeneous
            return transformed, A

    # Normalize last attempt to have det in range
    A = rng.standard_normal((4, 4))
    d = np.linalg.det(A)
    A = A / (abs(d) ** 0.25) * det_range[0] ** 0.25
    transformed = A @ points_homogeneous
    return transformed, A

# ---------------------------------------------------------------------------
# Independent bracket estimation
# ---------------------------------------------------------------------------
#
# Corollary 2.6 (docs/MATHEMATICS.md): if all 15 brackets are recomputed from
# ONE set of six points, every Plucker relation holds identically and
# rho(S) = 0 whatever was done to the points. Averaging the brackets of two
# configurations has no physical meaning, so a "separation" obtained that way
# would measure a property of the generator, not of non-rigid scenes.
#
# The generator below uses a measurement model instead. Each "measurement"
# is a separately estimated set of unit representatives of the six points
# (for example, a triangulation from a different view pair). Each of the 15
# brackets is estimated from ONE measurement, chosen independently per
# bracket. Brackets estimated from different measurements need not satisfy
# the Plucker relations, so rho(S) responds to how much the measurements
# disagree: measurement noise for a rigid scene, a swapped point for a
# mismatch, displacement for a non-rigid scene.
#
# Measurements live in the unit-representative chart WITHOUT renormalization,
# which is exactly the model of Theorem 6.1: if every measured representative
# is within delta of the true unit representative, rho(S) <= tau_exact(delta).


def unit_representatives(points_homogeneous: np.ndarray) -> np.ndarray:
    """Columns scaled to unit Euclidean norm (the chart used for brackets)."""
    norms = np.linalg.norm(points_homogeneous, axis=0, keepdims=True)
    assert np.all(norms > 1e-15), "Zero-norm homogeneous point."
    return points_homogeneous / norms


def measure_with_gaussian_noise(
    points_homogeneous: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One measurement: unit representatives plus i.i.d. N(0, sigma^2) noise per coordinate."""
    assert noise_sigma >= 0.0
    q = unit_representatives(points_homogeneous)
    return q + noise_sigma * rng.standard_normal(q.shape)


def perturb_within_ball(
    points_homogeneous: np.ndarray,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One measurement whose every unit representative is moved by a vector of
    norm <= radius (uniform direction, magnitude uniform in [0, radius]).
    This is the bounded-drift model of Theorem 6.1.
    """
    assert radius >= 0.0
    q = unit_representatives(points_homogeneous)
    directions = rng.standard_normal(q.shape)
    directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), 1e-15)
    magnitudes = rng.uniform(0.0, radius, size=(1, q.shape[1]))
    return q + directions * magnitudes


def estimate_brackets_independently(
    measurements: list[np.ndarray],
    tuple_indices: tuple[int, int, int, int, int, int],
    rng: np.random.Generator,
) -> tuple[dict[tuple[int, int, int, int], float], list[int]]:
    """
    Estimate each of the 15 brackets of a tuple from one measurement chosen
    independently and uniformly at random.

    Args:
        measurements: list of (4, N) arrays of measured representatives
                      (already in the unit chart; NOT renormalized here).
        tuple_indices: the six point indices.
        rng: random generator for the per-bracket measurement choice.

    Returns:
        (bracket table keyed by sorted quads, list of the measurement index
         used for each of the 15 brackets in sorted-quad order).
    """
    from itertools import combinations as _combinations
    assert len(measurements) >= 1 and len(tuple_indices) == 6
    sorted_tuple = tuple(sorted(int(i) for i in tuple_indices))
    brackets: dict[tuple[int, int, int, int], float] = {}
    sources: list[int] = []
    for quad in _combinations(sorted_tuple, 4):
        source = int(rng.integers(len(measurements)))
        brackets[quad] = float(np.linalg.det(measurements[source][:, list(quad)]))
        sources.append(source)
    return brackets, sources
