"""
ml_models/confidence_fusion/logistic_fusion.py

logistic Confidence Fusion (logistic fusion experiment; pre-registration docs/prereg/logistic_fusion.md).

Scope. Results apply to synthetic data with A and B given in a
common, known unit chart; the primary-arm inputs are not transferable to a
real reconstruction as specified. The logistic fusion is opt-in only: nothing in pipeline/ calls
this module, and LogisticFusionModel must be constructed explicitly.

Target. The label is the ablation study regime-mechanism label "tuple is bad"
(run_ablation.py), with the plane offset h and the non-rigid amplitude b
varied (pre-registration Section 3). A calibrated P(bad) therefore holds only
for that regime mixture and its prevalence; it is not a probability for any
other population.

Inputs. The neural fusion is fed rho from one point set, which is identically zero
(docs/MATHEMATICS.md, Corollary 2.6), and min(1, L/1000), which is saturated
at 1. the logistic fusion uses five log10 inputs, each defined and varying:
    kappa_J(A), L(A), sigma_4/sigma_1(A)  chart-dependent; computed in the
        generator's unit chart, where Proposition 5.1 applies (isotropic
        noise of known sigma on unit representatives).
    Q_0.1 of the 45 pencil discrepancies D_t between A and B  frame-free
        (Lemma 2.1: each pencil cross-ratio is gauge-invariant).
    rho under independent estimation (each bracket from A or B at random)
        not gauge-invariant when A and B have independent gauges (Eq. 1.1);
        informative only with A and B in one common chart with common unit
        representatives (the hypothesis of Theorem 6.1).
Arm D (deployable inputs) replaces the first three by the same scores of the
Tyler-whitened gauged measurement and drops rho.

Hard floor (CODEME section 7). An item is rejected, whatever the learned
score, if on A or on B: (stratum; algebraic) the rank stratum of the
Tyler-whitened points is CRITICAL; (undefined; algebraic) the invariants are
undefined or an input is non-finite; (conditioning; noise-model-dependent)
sigma * kappa_J >= 1. The conditioning constant is a convention motivated by
Proposition 5.1 (the point where the first-order SD of the worst log-invariant
reaches 1); it is not derived from that first-order result and depends on the
known noise model. decide_reject() returns floor OR (P >= threshold), so a
learned score can only add rejections, never remove one. Caveat for arm D:
evaluate_six_point_invariants uses an absolute denominator tolerance on unit
columns, so its None test on gauged measurements is not frame-free; the
rejections it causes are counted in experiments/results/logistic_fusion_counts.csv.

Why a learned combination at all (CODEME section 7). The label is a union of
mechanisms (degeneracy OR inconsistency between A and B), a linear logit
cannot express an OR, and no single input sees every mechanism: the pencil
statistic and rho see inconsistency, while kappa_J, L and sigma_4/sigma_1 see
conditioning, and kappa_J can stay finite when a bracket that J1-J3 do not
use vanishes (Proposition 5.2, converse), which L sees (sigma_4/sigma_1 is
the smallest singular value of the whole 4 x 6 matrix, so it sees only
near-coplanarity of the whole six-tuple, not one vanishing bracket).
Whether a fixed closed form does as well was tested, not assumed: H2 of the
pre-registration compares M1 with the closed-form three-threshold OR gate C3
(fit_or_gate on kappa_J, Q_0.1 and rho). On the pre-registered logistic fusion experiment test
split, H1 and H2 were both supported (case A of Section 9; numbers in
experiments/results/logistic_fusion_hypotheses.csv and logistic_fusion_comparisons.csv), so a
learned combination is justified on this synthetic, common-chart benchmark,
and only there. The H2 result concerns that specific three-input gate; it
does not show that no closed form suffices. M2 (gradient boosting) had a
lower test MCC than M1 (H3, same files), so M1 is the model kept here. The
deployable arm D is descriptive only and does not justify switching the logistic fusion on:
with deployable inputs (independent gauges, no rho), M1-D was not shown to
beat the floor plus the single pencil input (C1*-D) or the two-input OR gate
(C3-D); the CIs of both differences include 0 (logistic_fusion_comparisons.csv, arm D).
Part of the primary-arm gain comes from rho, which is informative only through
the common chart: M1 beat its ablation M1-no-rho (logistic_fusion_hypotheses.csv, row
"rho"), so that part of the gain depends on the common chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from algorithms.conditioning.scores import (
    compute_conditioning_score, compute_invariant_noise_gain, compute_singular_value_conditioning,
)
from algorithms.consistency.pencil import pencil_discrepancies
from algorithms.degeneracy.stratification import DegeneracyStratum, classify_stratum_by_rank
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.utils import whiten_homogeneous_points

PRIMARY_FEATURES = ("log_kappa_J", "log_L", "log_sv_ratio", "log_pencil_q10", "log_rho_indep")
DEPLOYABLE_FEATURES = ("log_kappa_J_w", "log_L_w", "log_sv_ratio_w", "log_pencil_q10_d")
NEURAL_FEATURES = ("neural_cond_norm", "neural_rho_single", "neural_bracket_strength")
PENCIL_QUANTILE = 0.1
RHO_LOG_CLIP = 1e-12                 # rho is clipped here before log10 (pre-registration Section 4)
STRATUM_TOLERANCE = 1e-3             # default of run_capri_reliability_layer
CONDITIONING_FLOOR_PRODUCT = 1.0     # sigma * kappa_J >= 1 rejects (convention, Proposition 5.1)
THRESHOLD_QUANTILES = np.linspace(0.005, 0.995, 199)
GAUGE_LOG10_RANGE = 1.0              # singular values of A_X and |lambda_k| in 10^[-1, 1]


def compute_pencil_quantile(points_a: np.ndarray, points_b: np.ndarray, tuple_indices: tuple) -> float:
    """Q_0.1 of the finite pencil discrepancies D_t of one six-tuple (45 values; NaN if none)."""
    assert points_a.shape == points_b.shape and len(tuple_indices) == 6
    discrepancies = pencil_discrepancies(points_a[:, list(tuple_indices)], points_b[:, list(tuple_indices)])
    finite = discrepancies[np.isfinite(discrepancies)]
    return float(np.quantile(finite, PENCIL_QUANTILE)) if finite.size else float("nan")


def draw_haar_orthogonal(rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed 4x4 orthogonal matrix: QR of a standard normal matrix, diag(R) > 0."""
    q_factor, r_factor = np.linalg.qr(rng.standard_normal((4, 4)))
    return q_factor * np.sign(np.diag(r_factor))[None, :]


def draw_random_gauge(n_points: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    One gauge (A_X, lambda) of Eq. (1.1), pre-registration Section 5.1.
    Draw order: U, V, u (A_X = U diag(10^u) V), then the signs s_k, then w_k
    (lambda_k = s_k 10^w_k). cond(A_X) <= 100.
    """
    assert n_points >= 6
    left, right = draw_haar_orthogonal(rng), draw_haar_orthogonal(rng)
    log_singular = rng.uniform(-GAUGE_LOG10_RANGE, GAUGE_LOG10_RANGE, 4)
    frame_map = left @ np.diag(10.0 ** log_singular) @ right
    signs = np.where(rng.integers(0, 2, n_points) == 1, 1.0, -1.0)
    point_scales = signs * 10.0 ** rng.uniform(-GAUGE_LOG10_RANGE, GAUGE_LOG10_RANGE, n_points)
    return frame_map, point_scales


def apply_gauge(points_homogeneous: np.ndarray, frame_map: np.ndarray, point_scales: np.ndarray) -> np.ndarray:
    """X -> A_X X diag(lambda)."""
    assert frame_map.shape == (4, 4) and point_scales.shape == (points_homogeneous.shape[1],)
    return frame_map @ points_homogeneous * point_scales[None, :]


def compute_floor_parts(points_homogeneous: np.ndarray, whitened_points: np.ndarray, invariant_noise_gain: float,
                        tuple_indices: tuple, noise_sigma: float) -> dict[str, bool]:
    """
    The three parts of the hard floor for one measurement (Section 4.2).
    stratum: rank stratum of the whitened points is CRITICAL (algebraic);
    undefined: J1-J3 undefined or kappa_J non-finite (algebraic);
    conditioning: noise_sigma * kappa_J >= 1 (noise-model-dependent convention).
    """
    assert noise_sigma > 0 and len(tuple_indices) == 6
    stratum = classify_stratum_by_rank(whitened_points, tuple_indices, tolerance=STRATUM_TOLERANCE)
    undefined = evaluate_six_point_invariants(points_homogeneous, tuple_indices) is None
    undefined = undefined or not np.isfinite(invariant_noise_gain)
    conditioning = bool(np.isfinite(invariant_noise_gain)
                        and noise_sigma * invariant_noise_gain >= CONDITIONING_FLOOR_PRODUCT)
    return {"stratum": stratum == DegeneracyStratum.CRITICAL, "undefined": bool(undefined),
            "conditioning": conditioning}


def compute_conditioning_inputs(points_homogeneous: np.ndarray, tuple_indices: tuple) -> dict[str, float]:
    """kappa_J, L and sigma_4/sigma_1 of one tuple, in the frame of the given points."""
    return {"kappa_J": compute_invariant_noise_gain(points_homogeneous, tuple_indices),
            "L": compute_conditioning_score(points_homogeneous, tuple_indices),
            "sv_ratio": compute_singular_value_conditioning(points_homogeneous, tuple_indices)}


def whiten_measurement(points_homogeneous: np.ndarray) -> np.ndarray:
    """Tyler-whitened unit points (the frame of run_capri_reliability_layer)."""
    return whiten_homogeneous_points(points_homogeneous)


def log10_inputs(raw_values: np.ndarray, rho_column: int | None = None) -> np.ndarray:
    """log10 of the raw inputs; the rho column (if any) is clipped at RHO_LOG_CLIP first."""
    values = np.array(raw_values, dtype=float, copy=True)
    if rho_column is not None:
        values[:, rho_column] = np.maximum(values[:, rho_column], RHO_LOG_CLIP)
    with np.errstate(divide="ignore", invalid="ignore"):      # log10(0) = -inf is caught by the floor
        return np.log10(values)


def build_m1_pipeline() -> Pipeline:
    """M1: logistic regression on the degree-2 expansion of standardised inputs (Section 5)."""
    return Pipeline([("scale_inputs", StandardScaler()), ("expand", PolynomialFeatures(2)),
                     ("scale_terms", StandardScaler()),
                     ("logit", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000))])


def build_m1_linear_pipeline() -> Pipeline:
    """M1-lin: M1 without the polynomial expansion."""
    return Pipeline([("scale_inputs", StandardScaler()),
                     ("logit", LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000))])


def build_m2_classifier(random_state: int) -> HistGradientBoostingClassifier:
    """M2: gradient boosting, settings fixed in Section 5, no early stopping."""
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15,
                                          class_weight="balanced", early_stopping=False,
                                          random_state=random_state)


def fit_platt_map(decision_values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Unweighted 1-D logistic recalibration (C = 1e6) of decision_function values."""
    assert decision_values.ndim == 1 and decision_values.shape == labels.shape
    assert np.all(np.isfinite(decision_values)) and set(np.unique(labels)) == {0, 1}
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(decision_values[:, None], labels)
    return float(platt.coef_[0, 0]), float(platt.intercept_[0])


def platt_probability(decision_values: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Calibrated P(bad) = 1 / (1 + exp(-(slope * z + intercept)))."""
    return 1.0 / (1.0 + np.exp(-(slope * np.asarray(decision_values, dtype=float) + intercept)))


def decide_reject(floor_rejected: np.ndarray, probability_bad: np.ndarray, threshold: float) -> np.ndarray:
    """Reject iff the hard floor fired OR P(bad) >= threshold; the floor is never overridden."""
    floor_rejected = np.asarray(floor_rejected, dtype=bool)
    probability_bad = np.asarray(probability_bad, dtype=float)
    assert floor_rejected.shape == probability_bad.shape
    with np.errstate(invalid="ignore"):
        return floor_rejected | (probability_bad >= threshold)


def weighted_confusion(weights: np.ndarray, labels: np.ndarray, decisions: np.ndarray) -> tuple:
    """(tp, fp, fn, tn) per row of a (resamples x items) weight matrix; positive = bad = reject."""
    positive, rejected = labels.astype(bool), decisions.astype(bool)
    assert weights.shape[1] == labels.shape[0] == decisions.shape[0]
    return (weights @ (rejected & positive), weights @ (rejected & ~positive),
            weights @ (~rejected & positive), weights @ (~rejected & ~positive))


def weighted_mcc(weights: np.ndarray, labels: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    """Matthews correlation per weight row; 0 when a margin is empty."""
    tp, fp, fn, tn = weighted_confusion(weights, labels, decisions)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(denominator > 0, (tp * tn - fp * fn) / np.where(denominator > 0, denominator, 1.0), 0.0)


def weighted_balanced_accuracy(weights: np.ndarray, labels: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    """Mean of the true-positive and true-negative rates per weight row."""
    tp, fp, fn, tn = weighted_confusion(weights, labels, decisions)
    return 0.5 * (tp / np.maximum(tp + fn, 1e-300) + tn / np.maximum(tn + fp, 1e-300))


def weighted_brier(weights: np.ndarray, labels: np.ndarray, probability_bad: np.ndarray) -> np.ndarray:
    """Weighted mean squared error of P(bad)."""
    return (weights @ (probability_bad - labels) ** 2) / weights.sum(axis=1)


def weighted_ece(weights: np.ndarray, labels: np.ndarray, probability_bad: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Expected calibration error over n_bins equal-width bins of P(bad) (P = 1 in the last bin)."""
    assert np.all((probability_bad >= 0) & (probability_bad <= 1))
    bins = np.minimum((probability_bad * n_bins).astype(int), n_bins - 1)
    residual_by_bin = np.zeros((labels.shape[0], n_bins))
    residual_by_bin[np.arange(labels.shape[0]), bins] = labels - probability_bad
    return np.abs(weights @ residual_by_bin).sum(axis=1) / weights.sum(axis=1)


def weighted_auc_ap(weights: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ROC AUC (trapezoid over distinct score values) and average precision
    (sum over distinct thresholds of recall step x precision), per weight row;
    identical to sklearn's roc_auc_score / average_precision_score with
    sample weights. Scores may be +inf (floor items); NaN is not allowed.
    """
    assert not np.any(np.isnan(scores))
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order].astype(float)
    group_ends = np.r_[np.nonzero(sorted_scores[1:] != sorted_scores[:-1])[0], sorted_scores.size - 1]
    sorted_weights = weights[:, order]
    true_pos = np.cumsum(sorted_weights * sorted_labels, axis=1)[:, group_ends]
    false_pos = np.cumsum(sorted_weights * (1.0 - sorted_labels), axis=1)[:, group_ends]
    tpr = np.hstack([np.zeros((weights.shape[0], 1)), true_pos / true_pos[:, -1:]])
    fpr = np.hstack([np.zeros((weights.shape[0], 1)), false_pos / false_pos[:, -1:]])
    auc = np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2.0, axis=1)
    flagged = true_pos + false_pos
    precision = np.where(flagged > 0, true_pos / np.where(flagged > 0, flagged, 1.0), 0.0)
    average_precision = np.sum(np.diff(tpr, axis=1) * precision, axis=1)
    return auc, average_precision


def choose_probability_threshold(probability_bad: np.ndarray, floor_rejected: np.ndarray,
                                 labels: np.ndarray) -> float:
    """
    Threshold maximising the MCC of the full decision (floor OR P >= t) on the
    given (calibration) items. Candidates: the 0.5%..99.5% quantiles (step 0.5%)
    of P over the floor-passing items only; the first maximum in ascending
    threshold order wins.
    """
    passing = ~floor_rejected.astype(bool)
    assert passing.any() and np.all(np.isfinite(probability_bad[passing]))
    candidates = np.unique(np.quantile(probability_bad[passing], THRESHOLD_QUANTILES))
    decisions = np.array([decide_reject(floor_rejected, np.where(passing, probability_bad, 1.0), t)
                          for t in candidates])
    return float(candidates[int(np.argmax(_mcc_rows(decisions, labels)))])


def fit_or_gate(inputs: np.ndarray, labels: np.ndarray, floor_rejected: np.ndarray, quantile_levels: np.ndarray,
                signs: tuple[float, ...] = (1.0,)) -> dict:
    """
    Closed-form gate: reject iff floor OR any_j s * x_j >= theta_j (one shared sign s).
    Candidate theta_j: the given quantiles of s * x_j over the floor-passing
    items; the MCC of the full rule on all given items is maximised by
    exhaustive search. Tie-break: signs in the given order, then thresholds in
    lexicographic ascending order (first maximum wins).
    """
    assert inputs.ndim == 2 and inputs.shape[0] == labels.shape[0] == floor_rejected.shape[0]
    passing = ~floor_rejected.astype(bool)
    best = {"mcc": -np.inf}
    for sign in signs:
        signed = sign * inputs
        assert np.all(np.isfinite(signed[passing]))
        grids = [np.quantile(signed[passing, j], quantile_levels) for j in range(inputs.shape[1])]
        with np.errstate(invalid="ignore"):
            last_hits = signed[:, -1][None, :] >= grids[-1][:, None]
        for leading in product(*[range(len(g)) for g in grids[:-1]]):
            base = floor_rejected.astype(bool).copy()
            for j, k in enumerate(leading):
                with np.errstate(invalid="ignore"):
                    base |= signed[:, j] >= grids[j][k]
            mccs = _mcc_rows(base[None, :] | last_hits, labels)
            k_last = int(np.argmax(mccs))
            if mccs[k_last] > best["mcc"]:
                thresholds = [grids[j][k] for j, k in enumerate(leading)] + [grids[-1][k_last]]
                best = {"mcc": float(mccs[k_last]), "sign": sign, "thresholds": np.array(thresholds)}
    return best


def _mcc_rows(decisions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """MCC of each row of a (candidates x items) boolean decision matrix."""
    positive = labels.astype(bool)
    tp = (decisions & positive).sum(axis=1).astype(float)
    fp = (decisions & ~positive).sum(axis=1).astype(float)
    fn = positive.sum() - tp
    tn = (~positive).sum() - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(denominator > 0, (tp * tn - fp * fn) / np.where(denominator > 0, denominator, 1.0), 0.0)


def apply_or_gate(inputs: np.ndarray, floor_rejected: np.ndarray, gate: dict) -> np.ndarray:
    """Decision of a fitted gate: floor OR any_j sign * x_j >= theta_j."""
    with np.errstate(invalid="ignore"):
        hits = (gate["sign"] * inputs >= gate["thresholds"][None, :]).any(axis=1)
    return floor_rejected.astype(bool) | hits


def fit_calibrated_rule(model, inputs: np.ndarray, labels: np.ndarray, floor_rejected: np.ndarray,
                        train_mask: np.ndarray, calibration_mask: np.ndarray, test_mask: np.ndarray) -> dict:
    """
    Section 5: fit on the floor-passing train items; Platt-recalibrate on the
    floor-passing calibration items; choose the threshold on all calibration
    items (choose_probability_threshold). Floor items get P(bad) = 1 and are
    never shown to the model. Returns test-split probabilities and decisions.
    """
    floor_rejected = floor_rejected.astype(bool)
    passing = ~floor_rejected
    assert not np.any((train_mask & calibration_mask) | (train_mask & test_mask) | (calibration_mask & test_mask))
    model.fit(inputs[train_mask & passing], labels[train_mask & passing])
    calibration_rows = calibration_mask & passing
    platt = fit_platt_map(model.decision_function(inputs[calibration_rows]), labels[calibration_rows])

    def probability_on(mask: np.ndarray) -> np.ndarray:
        probability = np.ones(int(mask.sum()))
        probability[passing[mask]] = platt_probability(model.decision_function(inputs[mask & passing]), *platt)
        return probability

    threshold = choose_probability_threshold(probability_on(calibration_mask), floor_rejected[calibration_mask],
                                             labels[calibration_mask])
    test_probability = probability_on(test_mask)
    return {"model": model, "platt": platt, "threshold": threshold, "probability": test_probability,
            "score": test_probability, "decision": decide_reject(floor_rejected[test_mask], test_probability, threshold),
            "decision_at_half": decide_reject(floor_rejected[test_mask], test_probability, 0.5)}


def compute_rule_metrics(weights: np.ndarray, labels: np.ndarray, rule: dict) -> dict[str, np.ndarray]:
    """MCC and balanced accuracy always; ROC AUC / AP if the rule has a score; Brier / ECE if a probability."""
    metrics = {"mcc": weighted_mcc(weights, labels, rule["decision"]),
               "balanced_accuracy": weighted_balanced_accuracy(weights, labels, rule["decision"])}
    if rule.get("score") is not None:
        metrics["roc_auc"], metrics["average_precision"] = weighted_auc_ap(weights, labels, rule["score"])
    if rule.get("probability") is not None:
        metrics["brier"] = weighted_brier(weights, labels, rule["probability"])
        metrics["ece"] = weighted_ece(weights, labels, rule["probability"])
    return metrics


def m1_parameter_frame(pipeline: Pipeline, platt: tuple[float, float], threshold: float,
                       feature_names: tuple[str, ...] = PRIMARY_FEATURES) -> pd.DataFrame:
    """All parameters of a fitted M1 (standardisation, terms, coefficients, Platt map, threshold)."""
    scale_in, expand = pipeline.named_steps["scale_inputs"], pipeline.named_steps["expand"]
    scale_terms, logit = pipeline.named_steps["scale_terms"], pipeline.named_steps["logit"]
    assert len(feature_names) == scale_in.mean_.size
    rows = [("input_mean", j, float(v), feature_names[j]) for j, v in enumerate(scale_in.mean_)]
    rows += [("input_scale", j, float(v), feature_names[j]) for j, v in enumerate(scale_in.scale_)]
    terms = [" ".join(str(int(p)) for p in powers) for powers in expand.powers_]
    rows += [("term_mean", j, float(v), terms[j]) for j, v in enumerate(scale_terms.mean_)]
    rows += [("term_scale", j, float(v), terms[j]) for j, v in enumerate(scale_terms.scale_)]
    rows += [("term_coefficient", j, float(v), terms[j]) for j, v in enumerate(logit.coef_[0])]
    rows += [("intercept", 0, float(logit.intercept_[0]), ""), ("platt_slope", 0, platt[0], ""),
             ("platt_intercept", 0, platt[1], ""), ("threshold", 0, float(threshold), "")]
    return pd.DataFrame(rows, columns=["parameter", "index", "value", "detail"])


@dataclass
class LogisticFusionModel:
    """
    Opt-in M1 predictor rebuilt from a parameter frame (the M1 rows of
    experiments/results/logistic_fusion_model.csv); no coefficient is typed in code.
    Valid only for the primary-arm inputs and population (see module docstring).
    """
    parameters: pd.DataFrame

    def _values(self, name: str) -> np.ndarray:
        rows = self.parameters[self.parameters["parameter"] == name].sort_values("index")
        assert len(rows) > 0, f"missing parameter {name}"
        return rows["value"].to_numpy(dtype=float)

    def probability_bad(self, log_inputs: np.ndarray) -> np.ndarray:
        """Calibrated P(bad) for floor-passing items (rows of the log10 inputs, in the fitted order)."""
        assert log_inputs.ndim == 2 and log_inputs.shape[1] == self._values("input_mean").size
        standardised = (log_inputs - self._values("input_mean")) / self._values("input_scale")
        rows = self.parameters[self.parameters["parameter"] == "term_mean"].sort_values("index")
        powers = np.array([[int(p) for p in str(d).split()] for d in rows["detail"]])
        expanded = np.prod(standardised[:, None, :] ** powers[None, :, :], axis=2)
        logit = ((expanded - self._values("term_mean")) / self._values("term_scale")) @ \
            self._values("term_coefficient") + self._values("intercept")[0]
        return platt_probability(logit, self._values("platt_slope")[0], self._values("platt_intercept")[0])

    def decide_reject(self, log_inputs: np.ndarray, floor_rejected: np.ndarray) -> np.ndarray:
        """Reject iff floor OR P(bad) >= the calibration-chosen threshold."""
        floor_rejected = np.asarray(floor_rejected, dtype=bool)
        probability = np.ones(floor_rejected.shape)
        if (~floor_rejected).any():
            probability[~floor_rejected] = self.probability_bad(log_inputs[~floor_rejected])
        return decide_reject(floor_rejected, probability, self._values("threshold")[0])
