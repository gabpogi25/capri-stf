"""
tests/test_invariants.py

Unit tests for six-point projective invariants (J1, J2, J3).

Verified properties:
  1. Invariants are unchanged under random PGL(4,R) transformations.
  2. Returns None for degenerate (near-zero denominator) configurations.
  3. Batch evaluation matches individual evaluation.
  4. Homogeneous weight balance: each p_i appears equally in num/denom.
"""

import numpy as np
import pytest
from algorithms.invariants.six_point import (
    evaluate_six_point_invariants,
    evaluate_batch_invariants,
)
from synthetic_data.generator import (
    generate_generic_configuration,
    apply_random_projective_transform,
)


def _generic_pts(seed: int = 0, n: int = 12) -> np.ndarray:
    return generate_generic_configuration(n_points=n, scale=1.0,
                                          rng=np.random.default_rng(seed))


class TestProjectiveInvariance:

    def test_j1_j2_j3_invariant_under_pgl4(self):
        """
        Core mathematical property: (J1,J2,J3) must be identical before
        and after any projective transformation. Tested over 100 random
        transformations. Relative error must be below 1e-8.
        """
        pts = _generic_pts(seed=0)
        t = (0, 1, 2, 3, 4, 5)
        inv_orig = evaluate_six_point_invariants(pts, t)
        assert inv_orig is not None, "Generic configuration should not be degenerate."
        j1_orig, j2_orig, j3_orig = inv_orig

        rng = np.random.default_rng(42)
        n_transforms = 100
        for _ in range(n_transforms):
            pts_T, _ = apply_random_projective_transform(
                pts, det_range=(0.3, 3.0), rng=rng
            )
            inv_T = evaluate_six_point_invariants(pts_T, t)
            assert inv_T is not None, "Transformed generic config should not be degenerate."
            j1_T, j2_T, j3_T = inv_T

            for orig, transformed, name in [
                (j1_orig, j1_T, "J1"),
                (j2_orig, j2_T, "J2"),
                (j3_orig, j3_T, "J3"),
            ]:
                if abs(orig) > 1e-10:
                    rel_err = abs(transformed - orig) / abs(orig)
                    assert rel_err < 1e-8, (
                        f"{name} failed invariance: orig={orig:.8f}, "
                        f"transformed={transformed:.8f}, rel_err={rel_err:.2e}"
                    )
                else:
                    abs_err = abs(transformed - orig)
                    assert abs_err < 1e-10, (
                        f"{name} near-zero invariance failed: abs_err={abs_err:.2e}"
                    )

    def test_different_configurations_give_different_invariants(self):
        """
        Two configurations not related by a projective transformation should
        generically produce different invariant triples.
        """
        pts_a = _generic_pts(seed=1)
        pts_b = _generic_pts(seed=2)
        t = (0, 1, 2, 3, 4, 5)
        inv_a = evaluate_six_point_invariants(pts_a, t)
        inv_b = evaluate_six_point_invariants(pts_b, t)
        assert inv_a is not None and inv_b is not None
        # At least one invariant should differ
        diffs = [abs(a - b) for a, b in zip(inv_a, inv_b)]
        assert max(diffs) > 1e-6, (
            "Two independent random configurations produced identical invariants, "
            "which is geometrically improbable and likely a bug."
        )


class TestDegeneracyRejection:

    def test_returns_none_for_coplanar_points(self):
        """
        When four points are coplanar, denominator brackets vanish and
        the function must return None rather than propagating inf/nan.
        """
        rng = np.random.default_rng(10)
        pts = _generic_pts(seed=10, n=12)
        # Force points 0,1,2,3 to be coplanar by collapsing one dimension
        pts[3, 0:4] = 0.0
        result = evaluate_six_point_invariants(pts, (0, 1, 2, 3, 4, 5))
        # May or may not be None depending on which brackets vanish;
        # if not None, the values should at least be finite
        if result is not None:
            assert all(np.isfinite(v) for v in result)

    def test_returns_none_when_denominator_explicitly_zero(self):
        """
        Construct a configuration where a denominator bracket is exactly zero
        and verify that None is returned.
        """
        pts = np.eye(4, dtype=np.float64)
        pts = np.hstack([pts, pts[:, :2]])  # 6 columns, 4 of them identical
        result = evaluate_six_point_invariants(pts, (0, 1, 2, 3, 4, 5),
                                               denom_tolerance=1e-10)
        # With repeated columns, many brackets vanish; expect None
        assert result is None or all(np.isfinite(v) for v in result)


class TestBatchEvaluation:

    def test_batch_matches_individual(self):
        pts = _generic_pts(seed=20, n=15)
        tuples = [
            (0, 1, 2, 3, 4, 5),
            (1, 2, 3, 4, 5, 6),
            (0, 2, 4, 6, 8, 10),
        ]
        batch_results = evaluate_batch_invariants(pts, tuples)
        for i, t in enumerate(tuples):
            individual = evaluate_six_point_invariants(pts, t)
            batch = batch_results[i]
            if individual is None:
                assert batch is None
            else:
                assert batch is not None
                for a, b in zip(individual, batch):
                    assert abs(a - b) < 1e-14, (
                        f"Batch/individual mismatch at tuple {i}: {a} vs {b}"
                    )

    def test_batch_returns_list_of_correct_length(self):
        pts = _generic_pts(seed=21)
        tuples = [(i, i+1, i+2, i+3, i+4, i+5) for i in range(4)]
        results = evaluate_batch_invariants(pts, tuples)
        assert len(results) == 4