"""
experiments/run_logistic_fusion.py

logistic Confidence Fusion with informative inputs (pre-registration
docs/prereg/logistic_fusion.md; it governs every choice below).

Items: (scene, two noisy measurements A and B, one random six-tuple), regime
REGIMES[k mod 6] in the order of run_ablation.py (Generic,
Near-Degenerate, Coplanar, Mismatched, Non-Rigid, Sparse-View). Label "tuple is
bad" from generator ground truth (Section 3). Seeds S = get_experiment_seed(19):
train S + 50e6 + k (6,000), calibration S + 51e6 + k (3,000), test S + 52e6 + k
(9,000), paired bootstrap S + 53e6 (B = 2,000). One default_rng per item; draw
order: tuple, regime parameter, scene, measurement A, measurement B, bracket
sources, arm-D gauges. No library call below re-seeds from the item seed.

Default invocation regenerates all items, writes logistic_fusion_dataset.csv, re-reads it
and fits only from the re-read values (bit-for-bit round-trip asserted).
--reuse-dataset reads an existing file instead and labels every output "_reused".

Bootstrap CIs are conditional on the fitted models, Platt maps, thresholds and
gates: nothing is refitted inside the bootstrap, so training and calibration
variability is not covered.

Outputs (experiments/results/, all via save_results, seed 42019):
logistic_fusion_dataset.csv, logistic_fusion_metrics.csv, logistic_fusion_comparisons.csv,
logistic_fusion_hypotheses.csv, logistic_fusion_regimes.csv, logistic_fusion_model.csv, logistic_fusion_counts.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import experiment_utils
from experiment_utils import MEASUREMENT_NOISE_SIGMA, get_experiment_seed, random_six_tuple, save_results
from run_ablation import (ASSUMED_DEFORMATION_BOUND, N_EXACT_COPLANAR, NEAR_DEGENERATE_FRACTION, REGIMES,
                               _coplanar_scene)
from synthetic_data.generator import (
    estimate_brackets_independently, generate_generic_configuration, generate_mismatched_configuration,
    generate_near_degenerate_configuration, generate_non_rigid_configuration, generate_sparse_view_configuration,
    measure_with_gaussian_noise,
)
from algorithms.backend import backend_name
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.syzygy.residual import compute_syzygy_residual
from ml_models.confidence_fusion.features import build_fusion_features_neural
from ml_models.confidence_fusion import logistic_fusion as fv3

EXPERIMENT_ID = 19
SEED = get_experiment_seed(EXPERIMENT_ID)  # 42019
SPLITS = (("train", 50_000_000, 6000), ("calibration", 51_000_000, 3000), ("test", 52_000_000, 9000))
N_POINTS, N_POINTS_SPARSE = 12, 8
N_NEAR_COPLANAR = int(N_POINTS * NEAR_DEGENERATE_FRACTION)       # first 6 of 12
PLANE_OFFSET_LOG10_RANGE = (-6.0, -3.0)
NON_RIGID_AMPLITUDE_LOG10_RANGE = (float(np.log10(0.005)), float(np.log10(0.1)))
MISMATCH_FRACTION = 0.15
DATASET_FILE = "logistic_fusion_dataset.csv"
STRING_COLUMNS = ("split", "regime", "tuple")
N_BOOTSTRAP, BOOTSTRAP_CHUNK, BOOTSTRAP_OFFSET = 2000, 200, 53_000_000
C3_QUANTILES = np.linspace(0.5, 0.995, 40)
ROW10_SCOPE = ("(synthetic data only, with A and B given in a common, known unit chart; "   # prereg Section 9
               "not tested on real images or on independently gauged reconstructions)")
PRIMARY_RAW = ("kappa_J", "L", "sv_ratio", "pencil_q10", "rho_indep")
DEPLOYABLE_RAW = ("kappa_J_w", "L_w", "sv_ratio_w", "pencil_q10_d")


def label_tuple_bad(regime: str, tuple_indices: tuple, scene_a: np.ndarray, scene_b: np.ndarray,
                    corrupted: list[int]) -> bool:
    """Ground-truth label of Section 3 (generator membership; never computed from an input)."""
    assert len(tuple_indices) == 6 and regime in REGIMES
    members = set(int(i) for i in tuple_indices)
    if regime == "Near-Degenerate":
        return len(members & set(range(N_NEAR_COPLANAR))) >= 4
    if regime == "Coplanar":
        return len(members & set(range(N_EXACT_COPLANAR))) >= 4
    if regime == "Mismatched":
        return bool(members & set(int(i) for i in corrupted))
    if regime == "Non-Rigid":
        displacement = np.linalg.norm((scene_b - scene_a)[1:, list(tuple_indices)], axis=0)
        return bool(displacement.max() > ASSUMED_DEFORMATION_BOUND)
    return False


def generate_regime_scenes(regime: str, rng: np.random.Generator) -> tuple:
    """(scene_a, scene_b, corrupted indices, regime parameter h or b or NaN); parameter drawn first."""
    corrupted: list[int] = []
    parameter = float("nan")
    if regime == "Near-Degenerate":
        parameter = 10.0 ** rng.uniform(*PLANE_OFFSET_LOG10_RANGE)
        scene_a = scene_b = generate_near_degenerate_configuration(
            N_POINTS, coplanar_fraction=NEAR_DEGENERATE_FRACTION, plane_offset=parameter, scale=1.0, rng=rng)
    elif regime == "Non-Rigid":
        parameter = 10.0 ** rng.uniform(*NON_RIGID_AMPLITUDE_LOG10_RANGE)
        scene_a, scene_b, _ = generate_non_rigid_configuration(N_POINTS, deformation_bound=parameter,
                                                               scale=1.0, rng=rng)
    elif regime == "Coplanar":
        scene_a = scene_b = _coplanar_scene(rng)
    elif regime == "Mismatched":
        scene_a, scene_b, corrupted = generate_mismatched_configuration(N_POINTS, MISMATCH_FRACTION, 1.0, rng)
    elif regime == "Sparse-View":
        scene_a, _ = generate_sparse_view_configuration(N_POINTS_SPARSE, n_views=2, points_per_view=6,
                                                        scale=1.0, rng=rng)
        scene_b = scene_a
    else:
        scene_a = scene_b = generate_generic_configuration(N_POINTS, scale=1.0, rng=rng)
    return scene_a, scene_b, [int(i) for i in corrupted], parameter


def measurement_record(points: np.ndarray, whitened: np.ndarray, kappa_points: np.ndarray, tuple_indices: tuple,
                       prefix: str) -> dict:
    """Floor parts of one measurement (stratum on its whitened points, kappa_J in kappa_points' frame)."""
    kappa = fv3.compute_conditioning_inputs(kappa_points, tuple_indices)["kappa_J"]
    parts = fv3.compute_floor_parts(points, whitened, kappa, tuple_indices, MEASUREMENT_NOISE_SIGMA)
    return {f"{prefix}_{name}": int(flag) for name, flag in parts.items()} | {f"{prefix}_kappa": kappa}


def generate_item(split: str, item_index: int, item_seed: int) -> dict:
    """One item: label, primary, neural and arm-D inputs, and both floors (Sections 3-5.1)."""
    rng = np.random.default_rng(item_seed)
    regime = REGIMES[item_index % len(REGIMES)]
    t = random_six_tuple(N_POINTS_SPARSE if regime == "Sparse-View" else N_POINTS, rng)
    scene_a, scene_b, corrupted, parameter = generate_regime_scenes(regime, rng)
    meas_a = measure_with_gaussian_noise(scene_a, MEASUREMENT_NOISE_SIGMA, rng)
    meas_b = measure_with_gaussian_noise(scene_b, MEASUREMENT_NOISE_SIGMA, rng)
    brackets, _ = estimate_brackets_independently([meas_a, meas_b], t, rng)
    gauges = [fv3.draw_random_gauge(meas_a.shape[1], rng) for _ in range(2)]
    gauged_a, gauged_b = fv3.apply_gauge(meas_a, *gauges[0]), fv3.apply_gauge(meas_b, *gauges[1])
    white_a, white_b = fv3.whiten_measurement(gauged_a), fv3.whiten_measurement(gauged_b)
    conditioning = fv3.compute_conditioning_inputs(meas_a, t)
    conditioning_w = fv3.compute_conditioning_inputs(white_a, t)
    neural = build_fusion_features_neural(meas_a, t)
    record = {"split": split, "item_index": item_index, "item_seed": item_seed, "regime": regime,
              "tuple": "-".join(str(int(i)) for i in t), "regime_parameter": parameter,
              "label_bad": int(label_tuple_bad(regime, t, scene_a, scene_b, corrupted)),
              "kappa_J": conditioning["kappa_J"], "L": conditioning["L"], "sv_ratio": conditioning["sv_ratio"],
              "pencil_q10": fv3.compute_pencil_quantile(meas_a, meas_b, t),
              "rho_indep": compute_syzygy_residual(brackets),
              "neural_cond_norm": neural[0], "neural_rho_single": neural[1], "neural_bracket_strength": neural[2],
              "kappa_J_w": conditioning_w["kappa_J"], "L_w": conditioning_w["L"],
              "sv_ratio_w": conditioning_w["sv_ratio"],
              "pencil_q10_d": fv3.compute_pencil_quantile(gauged_a, gauged_b, t),
              "L_gauged_raw": compute_conditioning_score(gauged_a, t)}
    record |= measurement_record(meas_a, fv3.whiten_measurement(meas_a), meas_a, t, "floor_a")
    record |= measurement_record(meas_b, fv3.whiten_measurement(meas_b), meas_b, t, "floor_b")
    record |= measurement_record(gauged_a, white_a, white_a, t, "floord_a")
    record |= measurement_record(gauged_b, white_b, white_b, t, "floord_b")
    return record


def add_floor_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine per-measurement floor parts into the floor of each arm, with causes (Sections 4.2, 5.1)."""
    primary_logs = fv3.log10_inputs(frame[list(PRIMARY_RAW)].to_numpy(), 4)
    deploy_logs = fv3.log10_inputs(frame[list(DEPLOYABLE_RAW)].to_numpy())
    for arm, prefix, logs in (("floor", "floor", primary_logs), ("floor_d", "floord", deploy_logs)):
        for part in ("stratum", "undefined", "conditioning"):
            frame[f"{arm}_{part}"] = frame[f"{prefix}_a_{part}"] | frame[f"{prefix}_b_{part}"]
        frame[f"{arm}_undefined"] |= (~np.isfinite(logs).all(axis=1)).astype(int)
        frame[arm] = frame[f"{arm}_stratum"] | frame[f"{arm}_undefined"] | frame[f"{arm}_conditioning"]
    return frame


def generate_dataset(split_sizes: dict[str, int] | None = None) -> pd.DataFrame:
    """All items of all splits from their seeds (CODEME section 8: before any fitting)."""
    rows = []
    for split, offset, n_items in SPLITS:
        count = n_items if split_sizes is None else split_sizes[split]
        rows += [generate_item(split, k, SEED + offset + k) for k in range(count)]
    return add_floor_decisions(pd.DataFrame(rows))


def write_and_reload_dataset(frame: pd.DataFrame, filename: str, experiment_name: str) -> pd.DataFrame:
    """Write with save_results, re-read, and assert that every numeric column round-trips bit for bit."""
    path = save_results(frame, filename, seed=SEED, experiment_name=experiment_name)
    reread = pd.read_csv(path, comment="#", float_precision="round_trip")
    assert list(reread.columns) == list(frame.columns) and len(reread) == len(frame)
    for column in frame.columns:
        if column in STRING_COLUMNS:
            assert (reread[column].astype(str) == frame[column].astype(str)).all(), column
        else:
            assert np.array_equal(reread[column].to_numpy(dtype=float), frame[column].to_numpy(dtype=float),
                                  equal_nan=True), f"round-trip mismatch in {column}"
    return reread


def obtain_dataset(reuse: bool, split_sizes: dict[str, int] | None = None) -> pd.DataFrame:
    """Default: regenerate and overwrite (never read an old file). reuse=True: read the existing file."""
    if reuse:
        print("WARNING: --reuse-dataset: reading an existing logistic_fusion_dataset.csv; outputs are labelled _reused.")
        return pd.read_csv(experiment_utils.RESULTS_DIR / DATASET_FILE, comment="#", float_precision="round_trip")
    frame = generate_dataset(split_sizes)
    return write_and_reload_dataset(frame, DATASET_FILE, "logistic fusion experiment: logistic Confidence Fusion item table")


def dataset_counts(frame: pd.DataFrame) -> list[tuple]:
    """Floor rejections by cause, clipped rho, label prevalence, gauge check and neural-fusion input checks."""
    rows = [("backend", backend_name(), np.nan)]
    for split in ["all"] + [s for s, _, _ in SPLITS]:
        part = frame if split == "all" else frame[frame["split"] == split]
        rows.append(("n_items", split, len(part)))
        rows.append(("label_bad_fraction", split, part["label_bad"].mean()))
        for arm in ("floor", "floor_d"):
            for cause in ("stratum", "undefined", "conditioning"):
                rows.append((f"{arm}_{cause}", split, int(part[f"{arm}_{cause}"].sum())))
            rows.append((f"{arm}_any", split, int(part[arm].sum())))
            rows.append((f"{arm}_algebraic_any", split, int((part[f"{arm}_stratum"] | part[f"{arm}_undefined"]).sum())))
            rows.append((f"{arm}_rejected_good", split, int((part[arm] & (part["label_bad"] == 0)).sum())))
            rows.append((f"{arm}_rejected_bad", split, int((part[arm] & (part["label_bad"] == 1)).sum())))
        rows.append(("rho_indep_clipped", split, int((part["rho_indep"] < fv3.RHO_LOG_CLIP).sum())))
    for regime in REGIMES:
        part = frame[frame["regime"] == regime]
        rows += [("label_bad_fraction", regime, part["label_bad"].mean()),
                 ("floor_any", regime, int(part["floor"].sum())), ("floor_d_any", regime, int(part["floor_d"].sum()))]
    both = np.isfinite(np.log10(frame["pencil_q10"])) & np.isfinite(np.log10(frame["pencil_q10_d"]))
    rows.append(("gauge_check_max_abs_dlog10_pencil_q10", "all",
                 float(np.max(np.abs(np.log10(frame["pencil_q10_d"][both]) - np.log10(frame["pencil_q10"][both]))))))
    finite_l = np.isfinite(frame["L"]) & np.isfinite(frame["L_gauged_raw"])
    rows.append(("gauge_control_max_abs_dlog10_L_raw", "all",
                 float(np.max(np.abs(np.log10(frame["L_gauged_raw"][finite_l]) - np.log10(frame["L"][finite_l]))))))
    rows += [("neural_rho_single_max", "all", float(frame["neural_rho_single"].max())),
             ("neural_cond_norm_median", "all", float(frame["neural_cond_norm"].median()))]
    return rows


def fit_arm(arm: str, suffix: str, labels: np.ndarray, masks: tuple, floor: np.ndarray, inputs: np.ndarray,
            input_names: tuple, gate_columns: list[int], learned_specs: dict, model_rows: list) -> dict:
    """Fit every rule of one arm on train / calibration and return its test-split outputs (Section 5)."""
    train, calibration, test = masks
    gates_fit = train | calibration
    rules = {f"C0{suffix}": {"decision": floor[test]}}
    for name, (make_model, columns) in learned_specs.items():
        fitted = fv3.fit_calibrated_rule(make_model(), columns, labels, floor, train, calibration, test)
        rules[name], rules[f"{name}@0.5"] = fitted, {"decision": fitted["decision_at_half"]}
        model_rows += [(arm, name, "threshold", 0, fitted["threshold"], ""),
                       (arm, name, "platt_slope", 0, fitted["platt"][0], ""),
                       (arm, name, "platt_intercept", 0, fitted["platt"][1], "")]
        if name in ("M1", "M1-D"):
            frame = fv3.m1_parameter_frame(fitted["model"], fitted["platt"], fitted["threshold"],
                                           input_names if columns.shape[1] == len(input_names) else ())
            model_rows += [(arm, name, *row) for row in frame.itertuples(index=False) if row[0] != "threshold"
                           and not row[0].startswith("platt")]
    best_single = (-np.inf, "")
    for j, feature in enumerate(input_names):
        gate = fv3.fit_or_gate(inputs[gates_fit][:, [j]], labels[gates_fit], floor[gates_fit],
                               fv3.THRESHOLD_QUANTILES, signs=(1.0, -1.0))
        rules[f"C1{suffix}:{feature}"] = {
            "decision": fv3.apply_or_gate(inputs[test][:, [j]], floor[test], gate),
            "score": np.where(floor[test], np.inf, gate["sign"] * np.nan_to_num(inputs[test, j]))}
        model_rows += [(arm, f"C1{suffix}:{feature}", "sign", 0, gate["sign"], ""),
                       (arm, f"C1{suffix}:{feature}", "threshold", 0, gate["thresholds"][0], ""),
                       (arm, f"C1{suffix}:{feature}", "train_cal_mcc", 0, gate["mcc"], "")]
        best_single = max(best_single, (gate["mcc"], feature), key=lambda pair: pair[0])
    rules[f"C1*{suffix}"] = rules[f"C1{suffix}:{best_single[1]}"]
    model_rows.append((arm, f"C1*{suffix}", "chosen_input", 0, best_single[0], best_single[1]))
    gate = fv3.fit_or_gate(inputs[gates_fit][:, gate_columns], labels[gates_fit], floor[gates_fit], C3_QUANTILES)
    rules[f"C3{suffix}"] = {"decision": fv3.apply_or_gate(inputs[test][:, gate_columns], floor[test], gate)}
    model_rows += [(arm, f"C3{suffix}", "threshold", k, value, input_names[gate_columns[k]])
                   for k, value in enumerate(gate["thresholds"])]
    model_rows.append((arm, f"C3{suffix}", "train_cal_mcc", 0, gate["mcc"], ""))
    return rules


def bootstrap_metrics(arms: dict, labels: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Point estimates and B = 2,000 paired-bootstrap replicates (same resamples for every rule)."""
    n_items = labels.size
    point, replicates = {}, {}

    def collect(weights: np.ndarray, store: dict) -> None:
        for arm, (rules, floor_test) in arms.items():
            for subset, mask in (("all", np.ones(n_items, bool)), ("floor_pass", ~floor_test)):
                for rule_name, rule in rules.items():
                    for metric, values in fv3.compute_rule_metrics(weights * mask[None, :], labels, rule).items():
                        store.setdefault((arm, rule_name, subset, metric), []).append(values)

    collect(np.ones((1, n_items)), point)
    rng = np.random.default_rng(seed + BOOTSTRAP_OFFSET)
    offsets = np.arange(BOOTSTRAP_CHUNK)[:, None] * n_items
    for _ in range(N_BOOTSTRAP // BOOTSTRAP_CHUNK):
        resample = rng.integers(0, n_items, size=(BOOTSTRAP_CHUNK, n_items))
        weights = np.bincount((offsets + resample).ravel(), minlength=BOOTSTRAP_CHUNK * n_items)
        collect(weights.reshape(BOOTSTRAP_CHUNK, n_items).astype(float), replicates)
    return ({key: float(v[0][0]) for key, v in point.items()},
            {key: np.concatenate(v) for key, v in replicates.items()})


def interval(values: np.ndarray) -> tuple[float, float]:
    """95% percentile interval."""
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def comparison_rows(point: dict, replicates: dict, pairs: list[tuple]) -> list[tuple]:
    """Delta = reference - rule for every shared (subset, metric); paired over the same resamples."""
    rows = []
    for (arm_ref, ref), (arm, rule) in pairs:
        for (a, r, subset, metric) in [key for key in point if key[:2] == (arm_ref, ref)]:
            if (arm, rule, subset, metric) in point:
                delta = replicates[(a, r, subset, metric)] - replicates[(arm, rule, subset, metric)]
                rows.append((arm_ref, ref, arm, rule, subset, metric,
                             point[(a, r, subset, metric)] - point[(arm, rule, subset, metric)], *interval(delta)))
    return rows


def hypothesis_rows(comparisons: pd.DataFrame) -> pd.DataFrame:
    """H1 (intersection-union), H2, H3, arm D, and the GOAL row 10 case of Section 9."""
    def delta(arm_ref: str, ref: str, arm: str, rule: str) -> tuple:
        row = comparisons[(comparisons.reference_arm == arm_ref) & (comparisons.reference == ref) &
                          (comparisons.arm == arm) & (comparisons.rule == rule) &
                          (comparisons.subset == "all") & (comparisons.metric == "mcc")].iloc[0]
        return row.delta, row.ci_low, row.ci_high

    rows = []
    for tag, rule in (("H1a", "C0"), ("H1b", "C1*"), ("H1c", "C2")):
        d = delta("primary", "M1", "primary", rule)
        rows.append((tag, f"M1 - {rule}", *d, "holds" if d[1] > 0 else "fails"))
    h1 = all(r[-1] == "holds" for r in rows)
    rows.append(("H1", "all three components", np.nan, np.nan, np.nan, "supported" if h1 else "fails"))
    d = delta("primary", "M1", "primary", "C3")
    h2 = "supported" if d[1] > 0 else ("C3 better" if d[2] < 0 else "no difference shown")
    rows.append(("H2", "M1 - C3 (this 3-input OR gate)", *d, h2))
    d = delta("primary", "M1", "primary", "M2")
    rows.append(("H3", "M2 - M1", -d[0], -d[2], -d[1], "differs" if d[1] > 0 or d[2] < 0 else "no difference shown"))
    d = delta("D", "M1-D", "D", "C0-D")
    rows.append(("D", "M1-D - C0-D (descriptive)", *d, "benefit shown" if d[1] > 0 else "no benefit shown"))
    # Descriptive rows that the row 10 text must carry next to "D" (prereg D2.3); they decide nothing.
    for tag, (arm_ref, ref, arm, rule) in (("D-C1*", ("D", "M1-D", "D", "C1*-D")), ("D-C3", ("D", "M1-D", "D", "C3-D")),
                                           ("rho", ("primary", "M1", "primary", "M1-no-rho"))):
        d = delta(arm_ref, ref, arm, rule)
        rows.append((tag, f"{ref} - {rule} (descriptive)", *d, "difference shown" if d[1] > 0 or d[2] < 0
                     else "no difference shown"))
    case = "D" if rows[0][-1] == "fails" else ("C" if not h1 else ("A" if h2 == "supported" else "B"))
    status = {"A": "Met", "B": "Partly met", "C": "Partly met", "D": "Not met"}[case]
    rows.append(("GOAL row 10", f"case {case}", np.nan, np.nan, np.nan, f"{status} {ROW10_SCOPE}"))
    return pd.DataFrame(rows, columns=["hypothesis", "comparison", "delta_mcc", "ci_low", "ci_high", "decision"])


def regime_rows(arms: dict, labels: np.ndarray, regimes: np.ndarray) -> list[tuple]:
    """Acceptance of good and bad test items per regime, for every rule."""
    rows = []
    for arm, (rules, _) in arms.items():
        for rule_name, rule in rules.items():
            for regime in REGIMES:
                good, bad = (regimes == regime) & (labels == 0), (regimes == regime) & (labels == 1)
                accepted = ~rule["decision"].astype(bool)
                rows.append((arm, rule_name, regime, int(good.sum()), int(bad.sum()),
                             int((accepted & good).sum()), int((accepted & bad).sum())))
    return rows


def evaluate_all_arms(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Primary arm, operating-point variant F1 and arm D; metrics, comparisons, hypotheses, regimes, model."""
    labels = frame["label_bad"].to_numpy(dtype=int)
    masks = tuple((frame["split"] == split).to_numpy() for split, _, _ in SPLITS)
    primary = fv3.log10_inputs(frame[list(PRIMARY_RAW)].to_numpy(), 4)
    deployable = fv3.log10_inputs(frame[list(DEPLOYABLE_RAW)].to_numpy())
    neural = frame[list(fv3.NEURAL_FEATURES)].to_numpy(dtype=float)
    floor, floor_d = frame["floor"].to_numpy(bool), frame["floor_d"].to_numpy(bool)
    generic_train = masks[0] & (frame["regime"] == "Generic").to_numpy()
    kappa_gate = float(np.median(frame["kappa_J"].to_numpy()[generic_train]))
    floor_f1 = floor | (frame["kappa_J"].to_numpy() >= kappa_gate)
    m1, lin, m2 = fv3.build_m1_pipeline, fv3.build_m1_linear_pipeline, lambda: fv3.build_m2_classifier(seed)
    base = {"M1": (m1, primary), "M1-lin": (lin, primary), "M2": (m2, primary), "C2": (m1, neural)}
    specs = base | {"C2-lin": (lin, neural), "C2-norho": (m1, neural[:, [0, 2]]),
                    "M1-no-pencil": (m1, primary[:, [0, 1, 2, 4]]), "M1-no-rho": (m1, primary[:, :4]),
                    "M1-conditioning-only": (m1, primary[:, :3])}
    model_rows = [("F1", "kappa_gate", "generic_train_median_kappa_J", 0, kappa_gate, "")]
    test = masks[2]
    arms = {"primary": (fit_arm("primary", "", labels, masks, floor, primary, fv3.PRIMARY_FEATURES, [0, 3, 4],
                                specs, model_rows), floor[test]),
            "F1": (fit_arm("F1", "", labels, masks, floor_f1, primary, fv3.PRIMARY_FEATURES, [0, 3, 4],
                           base, model_rows), floor_f1[test]),
            "D": (fit_arm("D", "-D", labels, masks, floor_d, deployable, fv3.DEPLOYABLE_FEATURES, [0, 3],
                          {"M1-D": (m1, deployable), "M1-lin-D": (lin, deployable)}, model_rows), floor_d[test])}
    arms["primary"][0]["C0-raw"] = {"decision": np.zeros(int(test.sum()), bool), "score": np.zeros(int(test.sum()))}
    point, replicates = bootstrap_metrics(arms, labels[test], seed)
    metrics = pd.DataFrame([(*key, value, *interval(replicates[key])) for key, value in point.items()],
                           columns=["arm", "rule", "subset", "metric", "estimate", "ci_low", "ci_high"])
    pairs = [((arm, ref), (arm, rule)) for arm, ref in (("primary", "M1"), ("F1", "M1"), ("D", "M1-D"))
             for rule in arms[arm][0] if rule != ref]
    pairs += [(("primary", "M1"), ("D", "M1-D")), (("primary", "M1"), ("F1", "M1"))]
    comparisons = pd.DataFrame(comparison_rows(point, replicates, pairs),
                               columns=["reference_arm", "reference", "arm", "rule", "subset", "metric",
                                        "delta", "ci_low", "ci_high"])
    regimes = pd.DataFrame(regime_rows(arms, labels[test], frame["regime"].to_numpy()[test]),
                           columns=["arm", "rule", "regime", "n_good", "n_bad", "accepted_good", "accepted_bad"])
    return {"metrics": metrics, "comparisons": comparisons, "hypotheses": hypothesis_rows(comparisons),
            "regimes": regimes, "model": pd.DataFrame(model_rows, columns=["arm", "rule", "parameter", "index",
                                                                         "value", "detail"])}


def run_experiment(reuse: bool = False) -> None:
    suffix, tag = ("_reused", " (dataset reused)") if reuse else ("", "")
    start = time.time()
    frame = obtain_dataset(reuse)
    generation_seconds = time.time() - start
    print(f"  dataset: {len(frame)} items in {generation_seconds:.0f} s")
    outputs = evaluate_all_arms(frame, SEED)
    outputs["counts"] = pd.DataFrame(dataset_counts(frame), columns=["quantity", "scope", "value"])
    for name, table in outputs.items():
        save_results(table, f"logistic_fusion_{name}{suffix}.csv", seed=SEED,
                     experiment_name=f"logistic fusion experiment: logistic Confidence Fusion, {name}{tag}")
    print(outputs["hypotheses"].to_string(index=False))
    print(f"  total runtime {time.time() - start:.0f} s (not written to a result file, to keep re-runs byte-identical)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument("--reuse-dataset", action="store_true",
                        help="read an existing logistic_fusion_dataset.csv (never used for committed results)")
    run_experiment(reuse=parser.parse_args().reuse_dataset)
