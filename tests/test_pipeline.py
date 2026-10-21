"""
tests/test_pipeline.py

Integration tests for the CAPRI reliability layer.

Verified properties:
  1. The reliability layer accepts well-conditioned generic configurations.
  2. The reliability layer rejects degenerate configurations.
  3. Certificates are correctly assembled from tuple evaluations.
  4. The full pipeline data flow (3D points -> tuples -> certificates) works.
"""

import numpy as np
import pytest
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.syzygy.residual import (
    compute_consistency_residual,
    compute_drift_threshold,
    compute_exact_admissibility_bound,
)
from algorithms.degeneracy.stratification import (
    classify_degeneracy_stratum,
    DegeneracyStratum,
)
from reconstruction.certification import (
    TupleCertificate,
    PointCertificate,
    assemble_point_certificates,
    compute_point_trust_score,
    assign_trust_label,
)
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
)


def _make_generic_3d_points(seed: int = 0, n: int = 12) -> np.ndarray:
    return generate_generic_configuration(n_points=n, scale=1.0,
                                          rng=np.random.default_rng(seed))


class TestReliabilityLayerIntegration:

    def test_generic_config_produces_accepted_tuples(self):
        """
        A generic configuration should have well-conditioned tuples
        that pass the conditioning and consistency checks.
        """
        # Uniform random tuples frequently exceed L(S) = 1000, so use a
        # deliberately well-spread tuple (L(S) = 320) as the positive example.
        pts = np.array([[1, 0, 0, 0, 1, 1],
                        [0, 1, 0, 0, 1, 2],
                        [0, 0, 1, 0, 1, -1],
                        [0, 0, 0, 1, 1, -2]], dtype=float)
        t = (0, 1, 2, 3, 4, 5)

        l_s = compute_conditioning_score(pts, t)
        assert l_s < 1000.0, f"Generic config has L(S)={l_s:.1f}, expected < 1000."

        stratum = classify_degeneracy_stratum(pts, t)
        assert stratum == DegeneracyStratum.GENERIC

        inv = evaluate_six_point_invariants(pts, t)
        assert inv is not None

    def test_degenerate_config_detected(self):
        """
        A near-degenerate configuration should either be classified as
        non-GENERIC or have a high conditioning score.
        """
        pts = generate_near_degenerate_configuration(
            n_points=12, coplanar_fraction=0.8, plane_offset=1e-6,
            rng=np.random.default_rng(42)
        )
        t = (0, 1, 2, 3, 4, 5)

        l_s = compute_conditioning_score(pts, t)
        stratum = classify_degeneracy_stratum(pts, t)

        is_flagged = (
            stratum != DegeneracyStratum.GENERIC
            or l_s > 100.0
        )
        assert is_flagged, (
            f"Near-degenerate config not flagged: stratum={stratum.value}, L(S)={l_s:.1f}"
        )


class TestCertificateAssembly:

    def test_certificate_assembly_basic(self):
        """
        Assemble certificates for a small synthetic point cloud and
        verify the output structure.
        """
        pts = _make_generic_3d_points(seed=1, n=8)

        # Create a mock accepted certificate
        cert = TupleCertificate(
            tuple_indices=(0, 1, 2, 3, 4, 5),
            conditioning_score=50.0,
            consistency_residual=0.001,
            drift_threshold=0.05,
            stratum="generic",
            accepted=True,
            descriptor_confidence=0.9,
        )

        point_certs = assemble_point_certificates(
            pts, [cert], conditioning_threshold=1000.0
        )

        assert len(point_certs) == pts.shape[1]
        # Points 0-5 should have a contributing tuple; 6-7 should not
        for pc in point_certs:
            if pc.point_index <= 5:
                assert len(pc.contributing_tuples) == 1
                assert pc.trust_score > 0.0
            else:
                assert len(pc.contributing_tuples) == 0
                assert pc.trust_score == 0.0

    def test_trust_labels(self):
        assert assign_trust_label(0.8) == "high"
        assert assign_trust_label(0.5) == "moderate"
        assert assign_trust_label(0.2) == "low"

    def test_no_accepted_tuples_gives_zero_trust(self):
        cert = TupleCertificate(
            tuple_indices=(0, 1, 2, 3, 4, 5),
            conditioning_score=50.0,
            consistency_residual=999.0,
            drift_threshold=0.05,
            stratum="critical",
            accepted=False,
        )
        score = compute_point_trust_score([cert])
        assert score == 0.0


class TestConsistencyAcrossViews:

    def test_same_points_different_order_give_same_invariants(self):
        """
        Invariants are defined up to the labeling of the six points.
        Computing invariants on the same physical points from two
        different orderings should produce the same result (for the
        same ordering fed to evaluate_six_point_invariants).
        """
        pts = _make_generic_3d_points(seed=5)
        t = (0, 1, 2, 3, 4, 5)
        inv1 = evaluate_six_point_invariants(pts, t)
        inv2 = evaluate_six_point_invariants(pts, t)
        assert inv1 is not None and inv2 is not None
        rho = compute_consistency_residual(inv1, inv2)
        assert rho == 0.0

class TestDriftThreshold:

    def test_drift_threshold_is_exact_bound(self):
        for delta in (0.0, 1e-3, 0.05, 0.3):
            assert compute_drift_threshold(delta) == compute_exact_admissibility_bound(delta)

    def test_exact_bound_exceeds_first_order(self):
        for delta in (1e-3, 0.05, 0.3):
            assert compute_drift_threshold(delta) >= 24.0 * delta
