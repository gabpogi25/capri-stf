"""
reconstruction/triangulation.py

Projective triangulation from certified six-point correspondences.

Given certified correspondences across multiple views (pairs of 2D image
points that passed conditioning and syzygy checks), this module computes
their 3D projective positions via linear triangulation (DLT).

The output is a projective point cloud, accurate up to an unknown
projective transformation of P^3. This is the best achievable result
from purely projective data without metric calibration.

Reference:
    Hartley & Zisserman (2003), Section 12.2 (linear triangulation).
    See docs/MATHEMATICS.md.
"""

from __future__ import annotations
import numpy as np


def triangulate_point_dlt(
    image_points: np.ndarray,
    projection_matrices: np.ndarray
) -> np.ndarray:
    """
    Triangulate a single 3D point from N >= 2 image observations via DLT.

    Each view contributes 2 equations. The system AX = 0 is solved via
    SVD, taking the right singular vector corresponding to the smallest
    singular value.

    Args:
        image_points: (N, 2) array of 2D image coordinates (pixel or normalized).
        projection_matrices: (N, 3, 4) array of camera projection matrices P_i.

    Returns:
        (4,) homogeneous 3D point coordinate.
    """
    assert image_points.ndim == 2 and image_points.shape[1] == 2
    assert projection_matrices.ndim == 3 and projection_matrices.shape[1:] == (3, 4)
    n_views = image_points.shape[0]
    assert n_views >= 2, "At least 2 views required for triangulation."

    rows = []
    for i in range(n_views):
        x, y = image_points[i]
        P = projection_matrices[i]
        rows.append(x * P[2, :] - P[0, :])
        rows.append(y * P[2, :] - P[1, :])

    A = np.stack(rows, axis=0)  # (2N, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1, :]  # (4,) homogeneous
    return X


def triangulate_point_cloud(
    image_points_per_view: list[np.ndarray],
    projection_matrices: np.ndarray,
    point_indices_per_view: list[list[int]] = None
) -> np.ndarray:
    """
    Triangulate multiple 3D points from multi-view correspondences.

    Args:
        image_points_per_view: List of (M_i, 2) arrays, one per view.
        projection_matrices: (N_views, 3, 4) projection matrix array.
        point_indices_per_view: Optional list of lists. point_indices_per_view[i]
            gives which global 3D point indices are visible in view i.
            If None, assumes all views see the same set of points in order.

    Returns:
        (4, M) homogeneous 3D point cloud, where M is the number of unique points.
    """
    n_views = len(image_points_per_view)
    assert projection_matrices.shape[0] == n_views

    if point_indices_per_view is None:
        # Assume all views see all points in the same order
        n_points = image_points_per_view[0].shape[0]
        point_indices_per_view = [list(range(n_points))] * n_views
        all_point_indices = set(range(n_points))
    else:
        all_point_indices = set()
        for idx_list in point_indices_per_view:
            all_point_indices.update(idx_list)

    # Build per-point observation lists
    observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        p: [] for p in all_point_indices
    }
    for v_idx in range(n_views):
        P = projection_matrices[v_idx]
        for k, p_idx in enumerate(point_indices_per_view[v_idx]):
            if k < image_points_per_view[v_idx].shape[0]:
                obs = image_points_per_view[v_idx][k]
                observations[p_idx].append((obs, P))

    # Triangulate each point
    points_3d = []
    valid_indices = []
    for p_idx in sorted(all_point_indices):
        obs_list = observations[p_idx]
        if len(obs_list) < 2:
            continue
        img_pts = np.stack([o[0] for o in obs_list], axis=0)
        proj_mats = np.stack([o[1] for o in obs_list], axis=0)
        X = triangulate_point_dlt(img_pts, proj_mats)
        if abs(X[3]) > 1e-10:  # avoid points at infinity
            points_3d.append(X / X[3])  # normalize to affine
        else:
            points_3d.append(X)
        valid_indices.append(p_idx)

    if not points_3d:
        return np.zeros((4, 0))

    return np.stack(points_3d, axis=1)  # (4, M)


def estimate_projection_matrices_from_fundamental(
    F: np.ndarray,
    image_points_1: np.ndarray,
    image_points_2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a camera pair (P1, P2) from the fundamental matrix F.

    P1 is set to the canonical form [I | 0].
    P2 is recovered from the epipole of F.

    This yields a projective reconstruction up to a 4x4 projective ambiguity.
    No metric calibration is assumed or implied.

    Args:
        F: (3, 3) fundamental matrix.
        image_points_1: (N, 2) corresponding points in view 1.
        image_points_2: (N, 2) corresponding points in view 2.

    Returns:
        P1: (3, 4) canonical camera matrix.
        P2: (3, 4) second camera matrix.
    """
    assert F.shape == (3, 3)

    # P1 = [I | 0]
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    # Epipole e2: F^T e2 = 0
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1, :]
    e2 /= np.linalg.norm(e2[:2])  # normalize for stability

    # Skew-symmetric matrix [e2]_x
    e2_skew = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])

    # P2 = [[e2]_x F | e2]
    P2 = np.hstack([e2_skew @ F, e2.reshape(3, 1)])

    return P1, P2


def estimate_fundamental_matrix_eight_point(
    points1: np.ndarray,
    points2: np.ndarray
) -> np.ndarray:
    """
    Estimate the fundamental matrix from >= 8 point correspondences
    using the normalized eight-point algorithm.

    Args:
        points1: (N, 2) image points in view 1.
        points2: (N, 2) image points in view 2.

    Returns:
        F: (3, 3) fundamental matrix with rank 2.
    """
    assert points1.shape == points2.shape
    assert points1.shape[0] >= 8
    n = points1.shape[0]

    # Normalize points
    def normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centroid = pts.mean(axis=0)
        pts_c = pts - centroid
        scale = np.sqrt(2) / (np.mean(np.linalg.norm(pts_c, axis=1)) + 1e-10)
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        pts_h = np.hstack([pts, np.ones((n, 1))])
        pts_norm = (T @ pts_h.T).T
        return pts_norm[:, :2], T

    p1n, T1 = normalize(points1)
    p2n, T2 = normalize(points2)

    # Build the design matrix A
    x1, y1 = p1n[:, 0], p1n[:, 1]
    x2, y2 = p2n[:, 0], p2n[:, 1]
    A = np.stack([
        x2*x1, x2*y1, x2,
        y2*x1, y2*y1, y2,
        x1, y1, np.ones(n)
    ], axis=1)

    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt2 = np.linalg.svd(F_norm)
    S[2] = 0
    F_norm = U @ np.diag(S) @ Vt2

    # Denormalize
    F = T2.T @ F_norm @ T1
    return F / np.linalg.norm(F)