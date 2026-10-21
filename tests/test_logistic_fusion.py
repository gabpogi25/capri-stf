"""
Tests for logistic Confidence Fusion (ml_models/confidence_fusion/logistic_fusion.py) and the
logistic fusion experiment runner (experiments/run_logistic_fusion.py); pre-registration
docs/prereg/logistic_fusion.md, Section 11. Each test names the failure it detects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import experiment_utils  # noqa: E402
import run_logistic_fusion as runner  # noqa: E402
from algorithms.conditioning.scores import compute_conditioning_score  # noqa: E402
from algorithms.utils import embed_affine_to_homogeneous  # noqa: E402
from ml_models.confidence_fusion import logistic_fusion as fv3  # noqa: E402
from synthetic_data.generator import generate_generic_configuration, measure_with_gaussian_noise  # noqa: E402

SIGMA = experiment_utils.MEASUREMENT_NOISE_SIGMA
TUPLE = (0, 1, 2, 3, 4, 5)


def _plane_scene(n_on_plane: int, offset: float, rng: np.random.Generator) -> np.ndarray:
    """12 points; the first n_on_plane lie within offset of the plane z = 0.1."""
    points = rng.uniform(-1.0, 1.0, size=(12, 3))
    points[:n_on_plane, 2] = 0.1 + offset * rng.uniform(-1.0, 1.0, n_on_plane)
    return embed_affine_to_homogeneous(points)


def test_pencil_input_is_gauge_invariant_and_raw_L_is_not():
    """Lemma 2.1 for Q_0.1; failure control: raw L(S) is chart-dependent and must change."""
    rng = np.random.default_rng(7)
    scene = generate_generic_configuration(12, rng=rng)
    meas_a, meas_b = (measure_with_gaussian_noise(scene, SIGMA, rng) for _ in range(2))
    gauged_a = fv3.apply_gauge(meas_a, *fv3.draw_random_gauge(12, rng))
    gauged_b = fv3.apply_gauge(meas_b, *fv3.draw_random_gauge(12, rng))
    q_common = fv3.compute_pencil_quantile(meas_a, meas_b, TUPLE)
    q_gauged = fv3.compute_pencil_quantile(gauged_a, gauged_b, TUPLE)
    assert abs(np.log10(q_gauged) - np.log10(q_common)) < 1e-8
    l_change = abs(np.log10(compute_conditioning_score(gauged_a, TUPLE)) -
                   np.log10(compute_conditioning_score(meas_a, TUPLE)))
    assert l_change > 1e-3


def test_learned_score_never_overrides_the_floor():
    """CODEME section 7: a floor-rejected item that M1 scores at P(bad) ~ 0 is still rejected."""
    rng = np.random.default_rng(11)
    measurement = measure_with_gaussian_noise(_plane_scene(4, 1e-7, rng), SIGMA, rng)
    raw = fv3.compute_conditioning_inputs(measurement, TUPLE)
    parts = fv3.compute_floor_parts(measurement, fv3.whiten_measurement(measurement), raw["kappa_J"], TUPLE, SIGMA)
    assert parts["conditioning"] and np.isfinite(raw["kappa_J"])
    item = np.array([[np.log10(raw["kappa_J"]), np.log10(raw["L"]), np.log10(raw["sv_ratio"]), -3.0, -4.0]])
    # Adversarial training data: high kappa_J is labelled good, so the model scores the item as good.
    toy = rng.normal(loc=[2.0, 5.0, -0.7, -2.0, -3.0], scale=1.0, size=(400, 5))
    toy_labels = (toy[:, 0] < 2.0).astype(int)
    model = fv3.build_m1_pipeline().fit(toy, toy_labels)
    platt = fv3.fit_platt_map(model.decision_function(toy), toy_labels)
    probability = fv3.platt_probability(model.decision_function(item), *platt)
    assert probability[0] < 0.01
    assert fv3.decide_reject(np.array([True]), probability, 0.5)[0]
    predictor = fv3.LogisticFusionModel(fv3.m1_parameter_frame(model, platt, 0.5))
    assert np.allclose(predictor.probability_bad(item), probability, rtol=1e-10)
    assert predictor.decide_reject(item, np.array([True]))[0]
    assert not predictor.decide_reject(item, np.array([False]))[0]


def test_floor_items_are_never_fitted_and_get_probability_one():
    """
    fit_calibrated_rule must fit and Platt-map on floor-passing items only, and
    give floor items P = 1. The floor items here carry finite inputs with
    inverted labels, so fitting on them changes the model and fails the test.
    """
    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(600, 5))
    labels = (inputs[:, 3] + 0.3 * rng.normal(size=600) > 0.5).astype(int)
    floor = np.zeros(600, bool)
    floor[::7] = True
    labels[floor] = 1 - labels[floor]
    split = np.arange(600) % 3
    train, calibration, test = split == 0, split == 1, split == 2
    fitted = fv3.fit_calibrated_rule(fv3.build_m1_pipeline(), inputs, labels, floor, train, calibration, test)
    reference = fv3.build_m1_pipeline().fit(inputs[train & ~floor], labels[train & ~floor])
    assert np.allclose(fitted["model"].decision_function(inputs), reference.decision_function(inputs))
    rows = calibration & ~floor
    assert np.allclose(fitted["platt"], fv3.fit_platt_map(reference.decision_function(inputs[rows]), labels[rows]))
    test_floor = floor[test]
    assert np.all(fitted["probability"][test_floor] == 1.0)
    assert np.all(fitted["decision"][test_floor]) and np.all(fitted["decision_at_half"][test_floor])
    assert np.all(fitted["probability"][~test_floor] < 1.0)


@pytest.mark.parametrize("regime, tuple_indices, expected", [
    ("Near-Degenerate", (0, 1, 2, 3, 7, 8), True), ("Near-Degenerate", (0, 1, 2, 6, 7, 8), False),
    ("Near-Degenerate", (0, 1, 2, 5, 7, 8), True),
    ("Coplanar", (0, 1, 2, 3, 7, 8), True), ("Coplanar", (0, 1, 2, 5, 7, 8), False),
    ("Generic", (0, 1, 2, 3, 4, 5), False), ("Sparse-View", (0, 1, 2, 3, 4, 5), False),
])
def test_membership_labels(regime, tuple_indices, expected):
    """Section 3: >= 4 of the first 6 (Near-Degenerate) or of the first 5 (Coplanar) is bad."""
    scene = np.ones((4, 12))
    assert runner.label_tuple_bad(regime, tuple_indices, scene, scene, []) is expected


def test_mismatch_and_non_rigid_labels():
    """Section 3: a replaced point in the tuple is bad; displacement > 0.02 on a tuple point is bad."""
    scene = np.ones((4, 12))
    assert runner.label_tuple_bad("Mismatched", (0, 1, 2, 3, 4, 9), scene, scene, [9])
    assert not runner.label_tuple_bad("Mismatched", (0, 1, 2, 3, 4, 5), scene, scene, [9])
    moved = scene.copy()
    moved[1, 10] += 0.021
    assert runner.label_tuple_bad("Non-Rigid", (0, 1, 2, 3, 4, 10), scene, moved, [])
    assert not runner.label_tuple_bad("Non-Rigid", (0, 1, 2, 3, 4, 5), scene, moved, [])
    moved[1, 10] = scene[1, 10] + 0.019
    assert not runner.label_tuple_bad("Non-Rigid", (0, 1, 2, 3, 4, 10), scene, moved, [])


def test_exp9_coplanar_scene_is_pinned():
    """The imported ablation study _coplanar_scene: 12 points, first 5 exactly coplanar, 36 + 3 draws consumed."""
    rng, reference = np.random.default_rng(5), np.random.default_rng(5)
    scene = runner._coplanar_scene(rng)
    affine = reference.uniform(-1.0, 1.0, size=(12, 3))
    normal = reference.standard_normal(3)
    assert scene.shape == (4, 12) and np.allclose(scene[1:, 5:], affine[5:].T)
    singular = np.linalg.svd(np.vstack([scene[1:, :5] - scene[1:, :1]]), compute_uv=False)
    assert singular[2] / singular[0] < 1e-12
    assert rng.uniform() == reference.uniform() and normal.shape == (3,)


def test_weighted_metrics_match_sklearn():
    """The bootstrap metric engine equals sklearn on the expanded (resampled) data."""
    rng = np.random.default_rng(1)
    labels = (rng.uniform(size=300) < 0.3).astype(int)
    scores = np.round(rng.normal(size=300) + labels, 1)
    scores[:20] = np.inf
    decisions = scores > 0.5
    resample = rng.integers(0, 300, 300)
    weights = np.bincount(resample, minlength=300)[None, :].astype(float)
    finite_scores = np.where(np.isinf(scores), 1e9, scores)
    auc, ap = fv3.weighted_auc_ap(weights, labels, scores)
    assert np.isclose(auc[0], roc_auc_score(labels[resample], finite_scores[resample]))
    assert np.isclose(ap[0], average_precision_score(labels[resample], finite_scores[resample]))
    assert np.isclose(fv3.weighted_mcc(weights, labels, decisions)[0],
                      matthews_corrcoef(labels[resample], decisions[resample]))


def test_or_gate_tie_break_takes_lowest_threshold_and_positive_sign():
    """R5 tie-break: among equal-MCC thresholds the lowest wins, and sign +1 before -1."""
    inputs = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]])
    labels = np.array([0, 0, 0, 0, 1, 1])
    levels = np.array([0.0, 0.2, 0.7, 0.8, 1.0])           # candidates 0, 1, 6.5, 10, 11
    gate = fv3.fit_or_gate(inputs, labels, np.zeros(6, bool), levels, signs=(1.0, -1.0))
    assert gate["sign"] == 1.0 and gate["mcc"] == 1.0
    assert gate["thresholds"][0] == pytest.approx(6.5)      # 10 also gives MCC 1; the lower one wins


def test_or_gate_tie_break_prefers_positive_sign_and_lower_leading_threshold():
    """R5 tie-break across signs and across the leading input of a two-input gate (first maximum wins)."""
    inputs = np.arange(6.0)[:, None]
    labels = np.array([1, 0, 0, 0, 0, 1])                  # sign +1 (t = 5) and sign -1 (t = 0) tie
    gate = fv3.fit_or_gate(inputs, labels, np.zeros(6, bool), np.array([0.0, 0.5, 1.0]), signs=(1.0, -1.0))
    assert gate["sign"] == 1.0 and gate["thresholds"][0] == pytest.approx(5.0)
    two_inputs = np.repeat(np.arange(4.0)[:, None], 2, axis=1)   # grids (0, 1.5, 3) for both inputs
    gate = fv3.fit_or_gate(two_inputs, np.array([0, 0, 1, 1]), np.zeros(4, bool), np.array([0.0, 0.5, 1.0]))
    assert gate["mcc"] == 1.0                              # (1.5, 1.5), (1.5, 3) and (3, 1.5) all reach 1
    assert np.allclose(gate["thresholds"], [1.5, 1.5])


def _small_sizes() -> dict:
    return {"train": 3, "calibration": 2, "test": 3}


def test_default_path_regenerates_and_round_trips(monkeypatch, tmp_path):
    """Section 6: the default path ignores (overwrites) an existing dataset and re-reads bit for bit."""
    monkeypatch.setattr(experiment_utils, "RESULTS_DIR", tmp_path)
    stale = tmp_path / runner.DATASET_FILE
    stale.write_text("split,regime\nSTALE,STALE\n")
    frame = runner.obtain_dataset(reuse=False, split_sizes=_small_sizes())
    assert len(frame) == 8 and "STALE" not in stale.read_text()
    assert list(frame["item_seed"][:3]) == [runner.SEED + 50_000_000 + k for k in range(3)]
    assert list(frame["regime"][:3]) == list(runner.REGIMES[:3])


def test_round_trip_check_detects_lossy_writes(monkeypatch, tmp_path):
    """Failure control for the round-trip assertion: a lossy float format must be caught."""
    monkeypatch.setattr(experiment_utils, "RESULTS_DIR", tmp_path)
    frame = pd.DataFrame({"split": ["train"], "regime": ["Generic"], "tuple": ["0-1-2-3-4-5"],
                          "kappa_J": [1.0 / 3.0]})

    def lossy_save(df, filename, seed, experiment_name):
        path = tmp_path / filename
        df.to_csv(path, index=False, float_format="%.6g")
        return path

    monkeypatch.setattr(runner, "save_results", lossy_save)
    with pytest.raises(AssertionError):
        runner.write_and_reload_dataset(frame, "lossy.csv", "test")


def test_docstring_carries_required_phrases():
    """Section 11: scope, transfer and floor-convention statements are in the module docstring."""
    for phrase in ("common, known unit chart", "not transferable", "convention", "opt-in"):
        assert phrase in fv3.__doc__


def test_row10_status_carries_scope_and_deployable_rows():
    """Section 9 / D2: the row 10 status is never written without its scope; arm-D and rho rows are present."""
    keys = [("primary", "M1", "primary", r) for r in ("C0", "C1*", "C2", "C3", "M2", "M1-no-rho")]
    keys += [("D", "M1-D", "D", r) for r in ("C0-D", "C1*-D", "C3-D")]
    comparisons = pd.DataFrame([(*k, "all", "mcc", 0.1, 0.05, 0.15) for k in keys],
                               columns=["reference_arm", "reference", "arm", "rule", "subset", "metric",
                                        "delta", "ci_low", "ci_high"])
    table = runner.hypothesis_rows(comparisons).set_index("hypothesis")
    status = table.loc["GOAL row 10", "decision"]
    assert status.startswith("Met (synthetic data only") and "common, known unit chart" in status
    assert table.loc["GOAL row 10", "comparison"] == "case A"
    for tag in ("D-C1*", "D-C3", "rho"):
        assert table.loc[tag, "decision"] == "difference shown"
    comparisons.loc[comparisons.rule == "C0", ["ci_low"]] = -0.01
    assert runner.hypothesis_rows(comparisons).set_index("hypothesis").loc["GOAL row 10", "decision"].startswith(
        "Not met (synthetic data only")
