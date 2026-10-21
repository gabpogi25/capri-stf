"""
tests/test_brackets.py

Unit tests for bracket determinant computation.

Verified properties:
  1. Bracket of identity-like matrix is 1.
  2. Swapping two columns negates the bracket (antisymmetry).
  3. Bracket scales by det(A) under projective transformation.
  4. Bracket of linearly dependent points is 0.
"""

import numpy as np
import pytest
from algorithms.invariants.brackets import (
    compute_bracket,
    compute_all_brackets_for_tuple,
    get_bracket,
)


def make_test_points(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts_3d = rng.uniform(-1, 1, size=(20, 3))
    ones = np.ones((20, 1))
    return np.hstack([ones, pts_3d]).T  # (4, 20)


class TestComputeBracket:

    def test_bracket_identity_block(self):
        pts = np.eye(4)  # columns are basis vectors
        val = compute_bracket(pts, (0, 1, 2, 3))
        assert abs(val - 1.0) < 1e-12

    def test_bracket_antisymmetry(self):
        pts = make_test_points(seed=1)
        val_orig = compute_bracket(pts, (0, 1, 2, 3))
        val_swap = compute_bracket(pts, (1, 0, 2, 3))
        assert abs(val_orig + val_swap) < 1e-10

    def test_raw_bracket_scales_with_transform(self):
        """det(A P) = det(A) det(P) for raw (unnormalized) representatives."""
        pts = make_test_points(seed=2)
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, 4))
        det_A = np.linalg.det(A)
        pts_transformed = A @ pts

        val_orig = compute_bracket(pts, (0, 1, 2, 3), pre_normalized=True)
        val_transformed = compute_bracket(pts_transformed, (0, 1, 2, 3), pre_normalized=True)

        assert abs(val_transformed - det_A * val_orig) < 1e-9 * (abs(det_A * val_orig) + 1)

    def test_normalized_bracket_scales_with_transform_and_norms(self):
        """
        compute_bracket normalizes each column first, so under A normalized
        bracket picks up det(A) times the ratio of the column norms.
        """
        pts = make_test_points(seed=2)
        rng = np.random.default_rng(7)
        A = rng.standard_normal((4, 4))
        det_A = np.linalg.det(A)
        pts_transformed = A @ pts
        quad = (0, 1, 2, 3)
        norm_ratio = np.prod(
            np.linalg.norm(pts[:, quad], axis=0) / np.linalg.norm(pts_transformed[:, quad], axis=0)
        )
        val_orig = compute_bracket(pts, quad)
        val_transformed = compute_bracket(pts_transformed, quad)
        expected = det_A * norm_ratio * val_orig
        assert abs(val_transformed - expected) < 1e-9 * (abs(expected) + 1)

    def test_bracket_dependent_points_is_zero(self):
        pts = make_test_points(seed=3)
        # Make point 3 a linear combination of 0, 1, 2
        pts[:, 3] = 0.4 * pts[:, 0] + 0.3 * pts[:, 1] + 0.3 * pts[:, 2]
        val = compute_bracket(pts, (0, 1, 2, 3))
        assert abs(val) < 1e-10

    def test_bracket_requires_4_indices(self):
        pts = make_test_points(seed=4)
        with pytest.raises(AssertionError):
            compute_bracket(pts, (0, 1, 2))

    def test_bracket_requires_4xN_input(self):
        pts = np.zeros((3, 10))
        with pytest.raises(AssertionError):
            compute_bracket(pts, (0, 1, 2, 3))


class TestAllBracketsForTuple:

    def test_returns_15_brackets_for_six_tuple(self):
        pts = make_test_points(seed=5)
        t = (0, 1, 2, 3, 4, 5)
        brackets = compute_all_brackets_for_tuple(pts, t)
        assert len(brackets) == 15

    def test_bracket_keys_are_sorted(self):
        pts = make_test_points(seed=6)
        t = (2, 5, 8, 11, 14, 17)
        brackets = compute_all_brackets_for_tuple(pts, t)
        for key in brackets:
            assert key == tuple(sorted(key))


class TestGetBracket:

    def test_get_bracket_matches_direct_compute(self):
        pts = make_test_points(seed=10)
        t = (0, 1, 2, 3, 4, 5)
        brackets = compute_all_brackets_for_tuple(pts, t)
        val_direct = compute_bracket(pts, (0, 1, 2, 3))
        val_lookup = get_bracket(brackets, 0, 1, 2, 3)
        assert abs(val_direct - val_lookup) < 1e-12

    def test_get_bracket_antisymmetry_via_permutation(self):
        pts = make_test_points(seed=11)
        t = (0, 1, 2, 3, 4, 5)
        brackets = compute_all_brackets_for_tuple(pts, t)
        val_01 = get_bracket(brackets, 0, 1, 2, 3)
        val_10 = get_bracket(brackets, 1, 0, 2, 3)
        assert abs(val_01 + val_10) < 1e-12


class TestNormalizedBrackets:

    def test_bracket_invariant_under_coordinate_scaling(self):
        """
        Since compute_bracket normalizes to the unit sphere internally,
        scaling all coordinates by a constant should not change the
        bracket value.
        """
        pts = make_test_points(seed=20)
        val_original = compute_bracket(pts, (0, 1, 2, 3))

        for scale in [0.01, 0.5, 2.0, 100.0]:
            val_scaled = compute_bracket(scale * pts, (0, 1, 2, 3))
            assert abs(val_scaled - val_original) < 1e-10, (
                f"Bracket changed under scaling by {scale}: "
                f"original={val_original:.8f}, scaled={val_scaled:.8f}"
            )

    def test_bracket_on_pre_normalized_matches_auto(self):
        """
        Passing pre_normalized=True on already-normalized points should
        give the same result as the default auto-normalization path.
        """
        from algorithms.utils import normalize_to_unit_sphere
        pts = make_test_points(seed=21)
        pts_norm = normalize_to_unit_sphere(pts)

        val_auto = compute_bracket(pts, (0, 1, 2, 3))
        val_pre = compute_bracket(pts_norm, (0, 1, 2, 3), pre_normalized=True)
        assert abs(val_auto - val_pre) < 1e-14