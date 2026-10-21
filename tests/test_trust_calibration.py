"""
tests/test_trust_calibration.py

Tests for the optional calibrated trust mapping (reconstruction/certification.py,
docs/prereg/trust_calibrated.md, Section 11) and for the label and metric
code of experiments/run_trust_calibrated.py.

The default R(p) output is pinned to values derived by hand from the formula
R = mean_t [0.5 ell + 0.3 d + 0.2 desc] (renormalised to (0.5 ell + 0.2 desc) / 0.7
when rho is NaN), so a change of the default behaviour fails here even though
tests/test_pipeline.py only checks signs. No calibration knot is typed in: maps
are fitted in the test or read from the result file when it exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import experiment_utils  # noqa: E402
import run_trust_calibrated as exp11b  # noqa: E402
from algorithms.invariants.six_point import evaluate_six_point_invariants  # noqa: E402
from forensics.fusion.calibrate import Calibrator  # noqa: E402
from reconstruction.certification import (  # noqa: E402
    TRUST_DOMAIN, TRUST_INPUT_CONVENTION, TrustCalibration, TupleCertificate, assemble_point_certificates,
    compute_calibrated_trust, compute_descriptor_free_trust_score, export_certificates_to_dict,
    load_trust_calibration,
)

THRESHOLD = 35.8  # a kappa threshold for the hand-computed fixtures; maps carry their own threshold
MAP_PATH = experiment_utils.RESULTS_DIR / "trust_calibrated_calibration_map.csv"


def certificate(indices: tuple, kappa: float, accepted: bool = True, descriptor: float = 1.0,
                rho: float = float("nan"), tau: float = float("nan")) -> TupleCertificate:
    return TupleCertificate(tuple_indices=indices, conditioning_score=kappa, consistency_residual=rho,
                            drift_threshold=tau, stratum="generic", accepted=accepted, descriptor_confidence=descriptor)


def fixture_certificates() -> list[TupleCertificate]:
    """14 points. 0: two accepted tuples; 1-5: one with NaN rho and one rejected; 7-11: one with a finite drift
    term; 12: only a rejected tuple; 6 and 13: no tuple."""
    return [certificate((0, 1, 2, 3, 4, 5), 2.0, descriptor=0.6),
            certificate((0, 7, 8, 9, 10, 11), 20.0, descriptor=0.8, rho=0.01, tau=0.04),
            certificate((1, 2, 3, 4, 5, 12), 50.0, accepted=False, descriptor=0.9)]


def isotonic_map(threshold: float = THRESHOLD, n: int = 6000, seed: int = 0) -> tuple[TrustCalibration, Calibrator]:
    """Isotonic Calibrator fitted on tied synthetic scores (>= 1,000 per class) and the TrustCalibration of its knots."""
    rng = np.random.default_rng(seed)
    scores = np.round(rng.uniform(2 / 7, 1.0, n), 2)  # rounding creates tied training scores
    labels = (rng.uniform(size=n) < 0.2 + 0.6 * (scores - 2 / 7)).astype(int)
    calibrator = Calibrator.fit(scores, labels)
    assert calibrator.method == "isotonic"
    knots = TrustCalibration(tuple(calibrator.model.X_thresholds_), tuple(calibrator.model.y_thresholds_),
                             TRUST_INPUT_CONVENTION, "kappa", threshold, 0.1)
    return knots, calibrator


def write_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: pd.DataFrame) -> Path:
    """Write a map with experiment_utils.save_results (provenance header included) into tmp_path."""
    monkeypatch.setattr(experiment_utils, "RESULTS_DIR", tmp_path)
    return experiment_utils.save_results(table, "trust_calibrated_calibration_map.csv", seed=1, experiment_name="test map")


def map_table(**overrides: object) -> pd.DataFrame:
    knots, _ = isotonic_map()
    table = pd.DataFrame(dict(x=knots.knots_r0, y=knots.knots_probability, input_convention=TRUST_INPUT_CONVENTION,
                              conditioning_metric="kappa", conditioning_threshold=THRESHOLD, label_tolerance=0.1,
                              method="isotonic", domain=TRUST_DOMAIN))
    for column, value in overrides.items():
        table[column] = value
    return table


# --------------------------------------------------------------------------- default behaviour unchanged

def test_default_output_is_pinned_to_the_hand_formula() -> None:
    points = np.vstack([np.random.default_rng(1).normal(size=(3, 14)), np.ones((1, 14))])
    certs = assemble_point_certificates(points, fixture_certificates(), THRESHOLD)  # the pipeline's 3-argument call
    renormalised = (0.5 * (1 - 2 / 35.8) + 0.2 * 0.6) / 0.7                    # rho NaN: weights 5/7 and 2/7
    with_drift = 0.5 * (1 - 20 / 35.8) + 0.3 * (1 - 0.01 / 0.04) + 0.2 * 0.8   # finite rho: 0.5, 0.3, 0.2
    expected = [(renormalised + with_drift) / 2] + [renormalised] * 5 + [0.0] + [with_drift] * 5 + [0.0, 0.0]
    np.testing.assert_allclose([c.trust_score for c in certs], expected, rtol=0, atol=1e-15)
    assert [c.trust_label for c in certs] == ["high"] * 6 + ["low"] + ["moderate"] * 5 + ["low", "low"]
    assert all(c.calibrated_trust is None for c in certs)
    exported = export_certificates_to_dict(certs)
    assert exported["n_points"] == 14
    assert [list(e) for e in exported["points"]] == [
        ["index", "trust_score", "trust_label", "n_contributing_tuples", "n_accepted_tuples"]] * 14
    assert [e["trust_score"] for e in exported["points"]] == [round(v, 6) for v in expected]
    assert [(e["n_contributing_tuples"], e["n_accepted_tuples"]) for e in exported["points"]] == \
        [(2, 2)] + [(2, 1)] * 5 + [(0, 0)] + [(1, 1)] * 5 + [(1, 0), (0, 0)]


# --------------------------------------------------------------------------- R0 and the domain D

def test_descriptor_free_score_ignores_descriptors_and_matches_trust_at_descriptor_one() -> None:
    low = [certificate((0, 1, 2, 3, 4, 5), 10.0, descriptor=0.3), certificate((0, 1, 2, 3, 4, 6), 30.0, descriptor=0.4)]
    high = [certificate(c.tuple_indices, c.conditioning_score, descriptor=1.0) for c in low]
    r0_low, r0_high = compute_descriptor_free_trust_score(low, THRESHOLD), compute_descriptor_free_trust_score(high, THRESHOLD)
    assert r0_low == r0_high
    assert r0_low == pytest.approx((0.5 * np.mean([1 - 10 / 35.8, 1 - 30 / 35.8]) + 0.2) / 0.7, abs=1e-15)
    points = np.vstack([np.zeros((3, 7)), np.ones((1, 7))])
    assert assemble_point_certificates(points, high, THRESHOLD)[0].trust_score == pytest.approx(r0_high, abs=1e-15)
    assert compute_descriptor_free_trust_score([certificate((0, 1, 2, 3, 4, 5), 10.0, accepted=False)], THRESHOLD) is None


def test_calibrated_trust_is_none_outside_D_and_uses_R0_inside() -> None:
    knots, calibrator = isotonic_map()
    certs = [certificate((0, 1, 2, 3, 4, 5), 10.0, descriptor=0.3), certificate((0, 1, 2, 3, 4, 6), 50.0, accepted=False)]
    points = np.vstack([np.zeros((3, 8)), np.ones((1, 8))])
    out = assemble_point_certificates(points, certs, THRESHOLD, trust_calibration=knots)
    r0 = compute_descriptor_free_trust_score(certs[:1], THRESHOLD)
    for p in range(6):
        assert out[p].calibrated_trust == pytest.approx(float(calibrator.predict([r0])[0]), abs=1e-12)
    assert out[6].calibrated_trust is None and out[7].calibrated_trust is None  # rejected-only and no tuple
    assert out[0].trust_score != r0  # trust_score keeps the descriptor confidence 0.3
    exported = export_certificates_to_dict(out, include_calibrated=True)["points"]
    assert exported[6]["calibrated_trust"] is None and exported[0]["calibrated_trust"] == round(out[0].calibrated_trust, 6)
    assert "calibrated_trust" not in export_certificates_to_dict(out)["points"][0]


def test_refuses_threshold_mismatch_and_finite_drift_term() -> None:
    knots, _ = isotonic_map()
    points = np.vstack([np.zeros((3, 6)), np.ones((1, 6))])
    certs = [certificate((0, 1, 2, 3, 4, 5), 10.0)]
    with pytest.raises(ValueError, match="conditioning_threshold"):
        assemble_point_certificates(points, certs, 1000.0, trust_calibration=knots)
    assemble_point_certificates(points, certs, THRESHOLD * (1 + 1e-12), trust_calibration=knots)  # rounding passes
    with pytest.raises(ValueError, match="drift term"):
        assemble_point_certificates(points, [certificate((0, 1, 2, 3, 4, 5), 10.0, rho=0.0, tau=0.1)], THRESHOLD,
                                    trust_calibration=knots)
    with pytest.raises(ValueError, match="drift term"):
        compute_calibrated_trust([certificate((0, 1, 2, 3, 4, 5), 10.0, rho=0.0, tau=0.1)], THRESHOLD, knots)


# --------------------------------------------------------------------------- the map and its loader

def test_predict_equals_sklearn_isotonic_including_ties_and_out_of_range() -> None:
    knots, calibrator = isotonic_map()
    queries = np.r_[np.random.default_rng(2).uniform(-0.5, 1.5, 5000), knots.knots_r0, [0.0, 2 / 7, 1.0, 5.0]]
    np.testing.assert_allclose(knots.predict(queries), calibrator.predict(queries), rtol=0, atol=1e-12)


def test_map_validation_rejects_bad_knots() -> None:
    for x, y in [((0.3, 0.3, 0.9), (0.1, 0.2, 0.3)), ((0.3, 0.6, 0.9), (0.2, 0.1, 0.3)),
                 ((0.3, 0.6, 0.9), (0.1, 0.2, 1.2)), ((0.3, np.nan, 0.9), (0.1, 0.2, 0.3))]:
        with pytest.raises(ValueError):
            TrustCalibration(x, y, TRUST_INPUT_CONVENTION, "kappa", THRESHOLD, 0.1)
    with pytest.raises(ValueError, match="input_convention"):
        TrustCalibration((0.3, 0.9), (0.1, 0.3), "trust_score_with_descriptors", "kappa", THRESHOLD, 0.1)


def test_loader_reads_save_results_file_and_keeps_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_map(tmp_path, monkeypatch, map_table())
    assert path.read_text().startswith("# Experiment: test map")
    loaded = load_trust_calibration(path, expected_threshold=THRESHOLD)
    knots, _ = isotonic_map()
    assert loaded.knots_r0 == pytest.approx(knots.knots_r0) and loaded.knots_probability == pytest.approx(knots.knots_probability)
    assert loaded.source_experiment == "test map" and loaded.source_commit != "unknown" and loaded.method == "isotonic"


@pytest.mark.parametrize("overrides, expected_threshold, expected_metric, message", [
    ({}, 35.0, "kappa", "conditioning_threshold"),
    ({"conditioning_metric": "L"}, THRESHOLD, "kappa", "conditioning_metric"),
    ({}, THRESHOLD, "L", "conditioning_metric"),
    ({"input_convention": "trust_score_with_descriptors"}, THRESHOLD, "kappa", "input_convention"),
    ({"domain": "all points"}, THRESHOLD, "kappa", "domain"),
    ({"two_values": "label_tolerance"}, THRESHOLD, "kappa", "more than one value"),
])
def test_loader_rejects_wrong_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict,
                                       expected_threshold: float, expected_metric: str, message: str) -> None:
    column = overrides.pop("two_values", None)
    table = map_table(**overrides)
    if column is not None:  # a metadata column holding two values
        table.loc[0, column] = 0.2
    path = write_map(tmp_path, monkeypatch, table)
    with pytest.raises(ValueError, match=message):
        load_trust_calibration(path, expected_threshold=expected_threshold, expected_metric=expected_metric)


def test_shipped_map_if_present() -> None:
    if not MAP_PATH.exists():
        pytest.skip("trust_calibrated_calibration_map.csv not generated yet (run experiments/run_trust_calibrated.py)")
    from pipeline.main_pipeline import calibrated_conditioning_threshold
    loaded = load_trust_calibration(MAP_PATH, expected_threshold=calibrated_conditioning_threshold(metric="kappa"))
    assert np.all(np.diff(loaded.knots_r0) > 0) and np.all(np.diff(loaded.knots_probability) >= 0)
    assert 0.0 <= min(loaded.knots_probability) and max(loaded.knots_probability) <= 1.0
    assert loaded.source_experiment.startswith("calibrated-trust experiment")


# --------------------------------------------------------------------------- label and metrics of calibrated-trust experiment

def test_invariant_error_is_frame_free_and_detects_moved_points() -> None:
    rng = np.random.default_rng(3)
    truth = np.vstack([rng.uniform(-1, 1, (3, 6)), np.ones((1, 6))])
    frame = rng.normal(size=(4, 4)) + 3 * np.eye(4)
    reconstructed = (frame @ truth) * rng.uniform(0.5, 2.0, 6) * np.sign(rng.normal(size=6))  # projective + rescaling
    eps, status = exp11b.tuple_invariant_error(reconstructed, truth, tuple(range(6)))
    assert status == "ok" and eps < 1e-9
    moved = truth.copy()
    moved[:3, 2] += 0.2
    eps_moved, status_moved = exp11b.tuple_invariant_error(moved, truth, tuple(range(6)))
    assert status_moved in ("ok", "sign_flip") and eps_moved > 1e-3
    mirrored = truth.copy()
    mirrored[:3, 0] = 2 * mirrored[:3, 1] - mirrored[:3, 0]  # reflect point 0 through point 1
    signs = [np.sign(evaluate_six_point_invariants(x, tuple(range(6)))) for x in (mirrored, truth)]
    assert np.any(signs[0] != signs[1]), "fixture must change a sign"
    assert exp11b.tuple_invariant_error(mirrored, truth, tuple(range(6))) == (np.inf, "sign_flip")


def test_weighted_metrics_match_reference_implementations() -> None:
    rng = np.random.default_rng(4)
    p = np.round(rng.uniform(size=400), 1)  # ties
    y = (rng.uniform(size=400) < p).astype(float)
    w = rng.integers(0, 4, 400).astype(float)
    score = exp11b.WeightedScore(p, y)
    assert score.auc(w) == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    bins = np.clip((p * 10).astype(int), 0, 9)
    manual = sum(abs(np.sum(w[bins == b] * (p - y)[bins == b])) for b in range(10)) / w.sum()
    assert score.ece(w) == pytest.approx(manual, abs=1e-12)
    assert score.brier(w) == pytest.approx(np.average((p - y) ** 2, weights=w), abs=1e-12)
    mass = score.mass_bins(np.ones(400))
    for value in np.unique(p):
        assert len(np.unique(mass[p == value])) == 1  # ties share an equal-mass bin
    constant = exp11b.WeightedScore(np.full(400, 0.3), y)
    assert constant.ece(w, equal_mass=True) == pytest.approx(abs(0.3 - np.average(y, weights=w)), abs=1e-12)


def test_tuple_selection_seed_is_decoupled_from_scene_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    real = exp11b.run_capri_reliability_layer

    def spy(*args: object, **kwargs: object) -> object:
        seen["seed"] = kwargs["seed"]
        return real(*args, **kwargs)
    monkeypatch.setattr(exp11b, "run_capri_reliability_layer", spy)
    scene_seed, threshold = 90_042_020, exp11b.calibrated_conditioning_threshold(metric="kappa")  # exploratory band
    rows, counts = exp11b.process_scene(scene_seed, False, threshold)
    assert seen["seed"] == scene_seed + exp11b.TUPLE_SEED_OFFSET != scene_seed
    assert counts["items"] == len(rows) > 0 and all(r["tuple_seed"] == seen["seed"] for r in rows)
    single = [r for r in rows if r["n_accepted_tuples"] == 1]  # R0 is a monotone transform of K here (Section 5)
    np.testing.assert_allclose([r["R0"] for r in single],
                               [(0.5 * (1 - 10 ** -r["K"] / threshold) + 0.2) / 0.7 for r in single], atol=1e-12)
