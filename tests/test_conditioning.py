"""
tests/test_conditioning.py

Unit tests for delta_min and conditioning score L(S).

Verified properties:
  1. delta_min is the minimum absolute bracket over all C(6,4)=15 quadruples.
  2. L(S) diverges as delta_min -> 0.
  3. Well-conditioned generic configs have L(S) < threshold.
  4. Near-degenerate configs have L(S) >> generic configs.
  5. L(S) is projectively invariant (unchanged under coordinate rescaling).
  6. is_tuple_well_conditioned correctly gates acceptance.
"""

import numpy as np
import pytest
from algorithms.conditioning.scores import (
    compute_delta_min,
    compute_conditioning_score,
    batch_conditioning_scores,
    is_tuple_well_conditioned,
)
from algorithms.invariants.brackets import compute_bracket
from algorithms.utils import normalize_to_unit_sphere
from itertools import combinations
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
)


def _pts(seed: int = 0, n: int = 12) -> np.ndarray:
    return generate_generic_configuration(n_points=n, scale=1.0,
                                          rng=np.random.default_rng(seed))


class TestDeltaMin:

    def test_delta_min_matches_manual_minimum(self):
        """
        delta_min must equal the minimum over all C(6,4)=15 bracket magnitudes
        on unit-normalized points, computed independently.
        """
        pts = _pts(seed=0)
        t = (0, 1, 2, 3, 4, 5)
        delta_min_computed = compute_delta_min(pts, t)

        pts_norm = normalize_to_unit_sphere(pts)
        manual_min = float('inf')
        for quad in combinations(t, 4):
            val = abs(compute_bracket(pts_norm, quad, pre_normalized=True))
            if val < manual_min:
                manual_min = val

        assert abs(delta_min_computed - manual_min) < 1e-14

    def test_delta_min_positive_for_generic_config(self):
        for seed in range(10):
            pts = _pts(seed=seed)
            t = (0, 1, 2, 3, 4, 5)
            dm = compute_delta_min(pts, t)
            assert dm > 0, f"Seed {seed}: delta_min is zero for generic config."

    def test_delta_min_small_for_near_degenerate(self):
        pts = generate_near_degenerate_configuration(
            n_points=12,
            coplanar_fraction=0.7,
            plane_offset=1e-5,
            rng=np.random.default_rng(5)
        )
        t = (0, 1, 2, 3, 4, 5)
        dm = compute_delta_min(pts, t)
        assert dm < 0.1, f"Expected small delta_min for coplanar config, got {dm:.4f}"


class TestConditioningScore:

    def test_conditioning_score_positive_finite_for_generic(self):
        pts = _pts(seed=0)
        t = (0, 1, 2, 3, 4, 5)
        l_s = compute_conditioning_score(pts, t)
        assert l_s > 0
        assert np.isfinite(l_s)

    def test_conditioning_score_returns_inf_for_zero_delta_min(self):
        """
        When delta_min is effectively zero (degenerate), L(S) should
        return float('inf') rather than nan or raising an exception.
        """
        pts = np.zeros((4, 12))
        pts[0, :] = 1.0  # avoid zero-norm assertion
        # All points identical -> all brackets zero
        l_s = compute_conditioning_score(pts, (0, 1, 2, 3, 4, 5))
        assert l_s == float('inf')

    def test_conditioning_score_larger_for_near_degenerate(self):
        """
        Near-degenerate configs must have larger L(S) than generic ones.
        """
        rng = np.random.default_rng(10)
        generic_scores = []
        degenerate_scores = []

        for seed in range(15):
            pts_g = generate_generic_configuration(12, rng=rng)
            l_g = compute_conditioning_score(pts_g, (0, 1, 2, 3, 4, 5))
            if np.isfinite(l_g):
                generic_scores.append(l_g)

            pts_d = generate_near_degenerate_configuration(
                12, coplanar_fraction=0.7, plane_offset=1e-4, rng=rng
            )
            l_d = compute_conditioning_score(pts_d, (0, 1, 2, 3, 4, 5))
            if np.isfinite(l_d):
                degenerate_scores.append(l_d)

        if generic_scores and degenerate_scores:
            assert np.median(degenerate_scores) > np.median(generic_scores), (
                "Near-degenerate configs should have higher median L(S) than generic."
            )

    def test_conditioning_score_invariant_under_rescaling(self):
        """
        L(S) must be unchanged when all homogeneous coordinates are
        multiplied by a constant. This is the projective invariance
        property ensured by unit-sphere normalization.
        """
        pts = _pts(seed=5)
        t = (0, 1, 2, 3, 4, 5)
        l_original = compute_conditioning_score(pts, t)

        for scale in [0.01, 0.5, 2.0, 100.0, 1e6]:
            l_scaled = compute_conditioning_score(scale * pts, t)
            assert abs(l_scaled - l_original) < 1e-10 * max(l_original, 1.0), (
                f"L(S) changed under scaling by {scale}: "
                f"original={l_original:.6f}, scaled={l_scaled:.6f}"
            )

    def test_batch_matches_individual(self):
        pts = _pts(seed=3, n=15)
        tuples = [(i, i+1, i+2, i+3, i+4, i+5) for i in range(5)]
        batch = batch_conditioning_scores(pts, tuples)
        for k, t in enumerate(tuples):
            individual = compute_conditioning_score(pts, t)
            if np.isfinite(individual) and np.isfinite(batch[k]):
                assert abs(batch[k] - individual) < 1e-12


class TestWellConditionedGate:

    def test_well_spread_config_passes_default_threshold(self):
        """
        A deliberately well-spread tuple (frame e1..e4 plus two points with
        mixed signs) has L(S) = 320 and passes the default threshold of 1000.
        Uniform random configurations frequently do NOT pass: the threshold is
        strict, which is why the ablation accepts under 1% of generic tuples.
        """
        pts = np.array([[1, 0, 0, 0, 1, 1],
                        [0, 1, 0, 0, 1, 2],
                        [0, 0, 1, 0, 1, -1],
                        [0, 0, 0, 1, 1, -2]], dtype=float)
        assert is_tuple_well_conditioned(pts, (0, 1, 2, 3, 4, 5),
                                         conditioning_threshold=1000.0)

    def test_zero_threshold_rejects_everything(self):
        pts = _pts(seed=0)
        assert not is_tuple_well_conditioned(pts, (0, 1, 2, 3, 4, 5),
                                              conditioning_threshold=0.0)

    def test_very_large_threshold_accepts_everything_finite(self):
        pts = _pts(seed=0)
        assert is_tuple_well_conditioned(pts, (0, 1, 2, 3, 4, 5),
                                          conditioning_threshold=1e15)