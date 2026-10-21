"""
experiments/run_trust_calibrated.py

calibrated-trust experiment: calibrated point trust scores (negative finding N7). Governed by the
pre-registration docs/prereg/trust_calibrated.md; section numbers below
refer to it.

Outcome (Section 3). Item = a reconstructed point p in D (survives RANSAC and
lies in at least one accepted tuple, |A(p)| >= 1). For an accepted tuple t,
eps(t) = max_k |ln|J_k(X^_t)| - ln|J_k(X_t)|| (natural log; X^ = reconstruction,
X = generator truth), +inf on a sign difference or an undefined reconstructed J.
Y(p) = 1 iff median_{t in A(p)} eps(t) <= 0.1. J1-J3 are pencil cross-ratios,
invariant with their sign under every linear change of frame and per-point
rescaling (docs/MATHEMATICS.md, Lemma 2.1), so the label needs no alignment.
Both clouds are whitened (Tyler) only for numerical conditioning of the
brackets: whitening is an invertible linear map, so it changes eps only by
rounding, also when Tyler's iteration stops before convergence.

What the calibrated output means. R-cal estimates P(Y = 1 | R0(p), p in D,
descriptor confidence 1, scene distribution of Section 4); it conditions on the
pipeline's acceptance (kappa_J gate at the calibrated threshold and greedy
coverage).

Why weak discrimination is expected. kappa_J in R0 is evaluated in the
whitened reconstruction frame, where pixel noise is anisotropic and of unknown
scale, and sigma_px varies tenfold across scenes. The label's tolerance is
absolute, and the error of a tuple is about sigma_px times its gain, so a
function of kappa_J alone cannot rank items across noise levels. Within one
reconstruction rho = 0 (Cor. 2.6), so R0 cannot see mismatches that lie on
their epipolar line either.

Why a fitted map (CODEME section 7). R(p) is a heuristic weighted mean; the
probability of the outcome at a given R depends on the scene distribution, so
no closed form maps R to a probability. The map is the isotonic Calibrator of
forensics/fusion/calibrate.py, fitted on the calibration split only.

Seeds (Section 6): S = get_experiment_seed(20) = 42020; scene k of a split uses
np.random.default_rng(scene_seed) for the scene, scene_seed + 500,000 for the
descriptor arm (draws U(0.25, 0.7) then U(0.4, 1.0) per inlier), scene_seed +
600,000 for tuple selection and cv2.setRNGSeed(scene_seed). The cluster
bootstrap (rng S + 73,000,000) draws the test resamples, then the shifted
resamples. The CIs hold the fitted maps fixed: they leave out the variance of
the calibration split (Deviations, item D1).

Equal-mass bins keep tied predictions in one bin (edges are weighted
quantiles of the predictions), so a constant or isotonic predictor is not
split at arbitrary item order (Deviations, item D2).

Outputs (save_results, seed S): trust_calibrated_items, _metrics, _comparisons,
_reliability, _strata, _counts, _calibration_map.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import get_experiment_seed, save_results
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.utils import embed_affine_to_homogeneous, whiten_homogeneous_points
from forensics.fusion.calibrate import Calibrator
from pipeline.main_pipeline import calibrated_conditioning_threshold, run_capri_reliability_layer
from reconstruction.certification import (TRUST_DOMAIN, TRUST_INPUT_CONVENTION, assemble_point_certificates,
                                          compute_descriptor_free_trust_score, load_trust_calibration)
from reconstruction.triangulation import estimate_projection_matrices_from_fundamental, triangulate_point_cloud

EXPERIMENT_SEED, EXPERIMENT_NAME = get_experiment_seed(20), "calibrated-trust experiment: calibrated trust (calibrated-trust pre-registration)"  # S = 42020
SPLITS = {"calibration": (70_000_000, 600, False), "test": (71_000_000, 1000, False),
          "shifted": (72_000_000, 300, True)}  # seed offset from S, scenes, shifted generator
DESCRIPTOR_SEED_OFFSET, TUPLE_SEED_OFFSET, BOOTSTRAP_SEED_OFFSET = 500_000, 600_000, 73_000_000
N_BOOTSTRAP, N_POINTS, MAX_TUPLES, N_BINS = 2000, 60, 200, 10
PRIMARY_TOLERANCE, SECONDARY_TOLERANCES = 0.1, (0.05, 0.2)
LABEL, PRIMARY = f"Y_tol{PRIMARY_TOLERANCE}", str(PRIMARY_TOLERANCE)
CURVE_SCORES = ("R_raw", "R_cal", "K_cal", "Base", "Platt_R")
S1_BOUND, S3_MARGIN, H2_LEVEL = 0.05, -0.01, 1 - 0.05 / 3
CAMERA_K = np.array([[800.0, 0, 1280 / 2], [0, 800.0, 960 / 2], [0, 0, 1.0]])  # f = 800 px, 1280 x 960 px
CAMERA_R = np.array([[np.cos(0.3), 0, np.sin(0.3)], [0, 1, 0], [-np.sin(0.3), 0, np.cos(0.3)]])
CAMERA_T = np.array([-2.0, 0.2, 0.5])
Writer = Callable[[pd.DataFrame, str], Path]
Statistic = Callable[[np.ndarray], "float | np.ndarray"]


# --------------------------------------------------------------------------- data (Sections 3-4)

def project_to_pixels(points_camera: np.ndarray) -> np.ndarray:
    projected = (CAMERA_K @ points_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def make_scene(rng: np.random.Generator, shifted: bool) -> dict:
    """Scene of Section 4 (draw order as listed there)."""
    planar_fraction = rng.uniform(0.0, 0.5)
    mismatch_fraction = rng.uniform(0.2, 0.35) if shifted else rng.uniform(0.0, 0.2)
    noise_px = 10 ** (rng.uniform(0.0, np.log10(5.0)) if shifted else rng.uniform(np.log10(0.3), np.log10(3.0)))
    n_planar = int(round(N_POINTS * planar_fraction))
    generic = np.column_stack([rng.uniform(-3, 3, N_POINTS - n_planar), rng.uniform(-2, 2, N_POINTS - n_planar),
                               rng.uniform(4, 12, N_POINTS - n_planar)])
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    basis_u = np.cross(normal, [0.0, 0.0, 1.0] if abs(normal[2]) < 0.9 else [1.0, 0.0, 0.0])
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    coefficients = rng.uniform(-3, 3, size=(n_planar, 2))
    planar = np.array([0.0, 0.0, 8.0]) + coefficients[:, :1] * basis_u + coefficients[:, 1:] * basis_v
    planar += np.outer(rng.uniform(-1e-3, 1e-3, n_planar) * 8.0, normal)
    permutation = rng.permutation(N_POINTS)
    world = np.vstack([generic, planar])[permutation]
    planar_mask = np.r_[np.zeros(N_POINTS - n_planar, bool), np.ones(n_planar, bool)][permutation]
    view_1 = project_to_pixels(world) + rng.normal(0, noise_px, (N_POINTS, 2))
    view_2 = project_to_pixels(world @ CAMERA_R.T + CAMERA_T) + rng.normal(0, noise_px, (N_POINTS, 2))
    mismatch = np.zeros(N_POINTS, bool)
    mismatch[rng.choice(N_POINTS, int(round(N_POINTS * mismatch_fraction)), replace=False)] = True
    for i in np.flatnonzero(mismatch):  # same camera-1 ray, other depth: on the true epipolar line
        moved = world[i] * rng.uniform(0.6, 1.6)
        view_2[i] = project_to_pixels((moved @ CAMERA_R.T + CAMERA_T)[None])[0] + rng.normal(0, noise_px, 2)
    return dict(world=world, view_1=view_1, view_2=view_2, planar=planar_mask, mismatch=mismatch,
                planar_fraction=planar_fraction, mismatch_fraction=mismatch_fraction, noise_px=noise_px)


def reconstruct_inliers(scene: dict, scene_seed: int) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """RANSAC F on raw pixels -> P1 = [I|0], P2 -> DLT points of the inliers (as main_pipeline)."""
    cv2.setRNGSeed(int(scene_seed))
    fundamental, mask = cv2.findFundamentalMat(scene["view_1"], scene["view_2"], cv2.FM_RANSAC, 3.0, 0.99)
    if fundamental is None or fundamental.shape != (3, 3) or mask is None:
        return None, None, "F_failed"
    inliers = np.flatnonzero(mask.ravel() == 1)
    if len(inliers) < 8:
        return None, None, "fewer_than_8_inliers"
    view_1, view_2 = scene["view_1"][inliers], scene["view_2"][inliers]
    camera_1, camera_2 = estimate_projection_matrices_from_fundamental(fundamental, view_1, view_2)
    return inliers, triangulate_point_cloud([view_1, view_2], np.stack([camera_1, camera_2])), ""


def tuple_invariant_error(reconstructed: np.ndarray, truth: np.ndarray, tuple_indices: tuple) -> tuple[float, str]:
    """eps(t) (natural log) and its status: ok, sign_flip, reconstructed_undefined or true_undefined."""
    def signed_invariants(points_homogeneous: np.ndarray) -> np.ndarray | None:
        invariants = evaluate_six_point_invariants(points_homogeneous, tuple_indices)
        invariants = None if invariants is None else np.asarray(invariants, float)
        return invariants if invariants is not None and np.all(np.isfinite(invariants) & (invariants != 0)) else None
    true_j, reconstructed_j = signed_invariants(truth), signed_invariants(reconstructed)
    if true_j is None:
        return float("nan"), "true_undefined"
    if reconstructed_j is None:
        return float("inf"), "reconstructed_undefined"
    if np.any(np.sign(reconstructed_j) != np.sign(true_j)):
        return float("inf"), "sign_flip"
    return float(np.max(np.abs(np.log(np.abs(reconstructed_j)) - np.log(np.abs(true_j))))), "ok"


def process_scene(scene_seed: int, shifted: bool, threshold: float) -> tuple[list[dict], dict]:
    """Items of one scene (points in D) and the scene's counts for trust_calibrated_counts."""
    scene = make_scene(np.random.default_rng(scene_seed), shifted)
    counts = dict(scene_seed=scene_seed, **dict.fromkeys(("ransac_surviving", "surviving_mismatches", "outside_D", "items",
                  "true_undefined_items", "single_tuple_items", "accepted_tuples", "sign_flip", "reconstructed_undefined",
                  "true_undefined"), 0))
    inliers, points, counts["excluded"] = reconstruct_inliers(scene, scene_seed)
    if inliers is None:
        return [], counts
    mismatch, tuple_seed = scene["mismatch"][inliers], scene_seed + TUPLE_SEED_OFFSET
    certificates, _ = run_capri_reliability_layer(points, np.ones(len(inliers)), conditioning_threshold=threshold,
                                                  conditioning_metric="kappa", max_tuples=MAX_TUPLES, seed=tuple_seed)
    point_certificates = assemble_point_certificates(points, certificates, threshold)
    descriptor_rng = np.random.default_rng(scene_seed + DESCRIPTOR_SEED_OFFSET)
    descriptor = np.where(mismatch, descriptor_rng.uniform(0.25, 0.7, len(inliers)),
                          descriptor_rng.uniform(0.4, 1.0, len(inliers)))
    with_descriptor = [dataclasses.replace(c, descriptor_confidence=float(np.mean(descriptor[list(c.tuple_indices)])))
                       for c in certificates]  # the pipeline's tuple descriptor confidence: mean over the tuple
    trust_descriptor = [c.trust_score for c in assemble_point_certificates(points, with_descriptor, threshold)]
    truth, reconstructed = map(whiten_homogeneous_points, (embed_affine_to_homogeneous(scene["world"][inliers]), points))
    errors = {c.tuple_indices: tuple_invariant_error(reconstructed, truth, c.tuple_indices)
              for c in certificates if c.accepted}
    statuses = [status for _, status in errors.values()]
    counts.update(ransac_surviving=len(inliers), surviving_mismatches=int(mismatch.sum()), accepted_tuples=len(errors),
                  **{s: statuses.count(s) for s in ("sign_flip", "reconstructed_undefined", "true_undefined")})
    rows = []
    for point in point_certificates:
        accepted = [c for c in point.contributing_tuples if c.accepted]
        undefined = any(errors[c.tuple_indices][1] == "true_undefined" for c in accepted)
        counts["outside_D"] += not accepted
        counts["true_undefined_items"] += bool(accepted) and undefined
        if not accepted or undefined:
            continue
        r0 = compute_descriptor_free_trust_score(accepted, threshold)
        assert abs(r0 - point.trust_score) < 1e-12, "R0 must equal trust_score at descriptor confidence 1"
        eps_median = float(np.median([errors[c.tuple_indices][0] for c in accepted]))
        others = {j for c in accepted for j in c.tuple_indices if j != point.point_index}
        rows.append(dict(scene_seed=scene_seed, tuple_seed=tuple_seed, point_index=point.point_index, R0=r0,
                         K=-float(np.median(np.log10([c.conditioning_score for c in accepted]))),
                         R_desc=trust_descriptor[point.point_index], n_accepted_tuples=len(accepted),
                         eps_median=eps_median, **{f"Y_tol{t}": int(eps_median <= t)
                                                   for t in (PRIMARY_TOLERANCE, *SECONDARY_TOLERANCES)},
                         mismatch=bool(mismatch[point.point_index]), other_tuple_mismatch=bool(any(mismatch[list(others)])),
                         planar=bool(scene["planar"][inliers[point.point_index]]), n_inliers=len(inliers),
                         **{k: scene[k] for k in ("planar_fraction", "mismatch_fraction", "noise_px")}))
    counts.update(items=len(rows), single_tuple_items=sum(r["n_accepted_tuples"] == 1 for r in rows))
    return rows, counts


def collect_split(split: str, seed_offset: int, n_scenes: int, shifted: bool, threshold: float,
                  seed_base: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = [process_scene(seed_base + seed_offset + k, shifted, threshold) for k in range(n_scenes)]
    items = pd.DataFrame([row for rows, _ in results for row in rows])
    items.insert(0, "split", split)
    return items, pd.DataFrame([counts for _, counts in results]).assign(split=split)


# --------------------------------------------------------------------------- weighted metrics
# A scene-bootstrap resample is a weight vector (multiplicity of each item's scene);
# weights of 1 give the point estimate.

class WeightedScore:
    """ECE (equal-width / equal-mass), Brier, AUC and reliability of predictions p for labels y."""

    def __init__(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        self.p, self.y = np.asarray(predictions, float), np.asarray(labels, float)
        assert self.p.shape == self.y.shape and np.all(np.isfinite(self.p)), "predictions must be finite"
        self.width_bins = np.clip((self.p * N_BINS).astype(int), 0, N_BINS - 1)
        self.order = np.argsort(self.p, kind="stable")
        self.group_starts = np.flatnonzero(np.r_[True, np.diff(self.p[self.order]) > 0])

    def mass_bins(self, weights: np.ndarray) -> np.ndarray:
        """Edges = weighted deciles of p; tied predictions always share a bin."""
        cumulative = np.cumsum(weights[self.order])
        positions = np.searchsorted(cumulative, cumulative[-1] * np.arange(1, N_BINS) / N_BINS, side="left")
        return np.searchsorted(self.p[self.order][np.minimum(positions, len(cumulative) - 1)], self.p, side="left")

    def ece(self, weights: np.ndarray, equal_mass: bool = False) -> float:
        bins = self.mass_bins(weights) if equal_mass else self.width_bins
        return float(np.abs(np.bincount(bins, weights * (self.p - self.y), minlength=N_BINS)).sum() / weights.sum())

    def brier(self, weights: np.ndarray) -> float:
        return float(np.sum(weights * (self.p - self.y) ** 2) / weights.sum())

    def auc(self, weights: np.ndarray) -> float:
        """Weighted Mann-Whitney AUC; ties count one half."""
        w, y = weights[self.order], self.y[self.order]
        positives, negatives = np.add.reduceat(w * y, self.group_starts), np.add.reduceat(w * (1 - y), self.group_starts)
        denominator, below = positives.sum() * negatives.sum(), np.cumsum(negatives) - negatives
        return float(np.sum(positives * (below + 0.5 * negatives)) / denominator) if denominator > 0 else float("nan")

    def reliability(self, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Per bin, stacked: weighted mean prediction, observed frequency, weight."""
        n = np.bincount(bins, weights, minlength=N_BINS)
        return np.r_[[np.bincount(bins, weights * v, minlength=N_BINS) / n for v in (self.p, self.y)], [n]].ravel()


def scene_codes(items: pd.DataFrame) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(items["scene_seed"], sort=True)
    return codes, len(uniques)


def bootstrap_counts(rng: np.random.Generator, n_scenes: int) -> np.ndarray:
    """Scene multiplicities of N_BOOTSTRAP resamples with replacement."""
    draws = rng.integers(0, n_scenes, size=(N_BOOTSTRAP, n_scenes))
    return np.stack([np.bincount(row, minlength=n_scenes) for row in draws]).astype(float)


def run_bootstrap(statistics: dict[str, Statistic], codes: np.ndarray, counts: np.ndarray) -> tuple[dict, dict]:
    """Point estimates and draws over the scene resamples (the same resamples for every statistic)."""
    with np.errstate(invalid="ignore", divide="ignore"):  # empty reliability bins give NaN, kept as NaN
        estimates = {name: np.asarray(f(np.ones(len(codes)))) for name, f in statistics.items()}
        draws = {name: np.empty((len(counts),) + estimates[name].shape) for name in statistics}
        for b, scene_counts in enumerate(counts):
            for name, f in statistics.items():
                draws[name][b] = f(scene_counts[codes])
    return estimates, draws


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval; NaN when no draw is finite (e.g. an empty reliability bin)."""
    tail, draws = 100 * (1 - level) / 2, np.asarray(draws, float)
    finite = np.any(np.isfinite(draws))
    return tuple(float(np.nanpercentile(draws, q)) if finite else float("nan") for q in (tail, 100 - tail))


# --------------------------------------------------------------------------- analysis (Sections 5, 7, 8)
# Statistic keys: "<tolerance>:<score>|<metric>", "strata:<stratum>|<metric>", "curve:<score>|<binning>".

def fit_score_maps(calibration: pd.DataFrame, label: str) -> dict:
    """Calibrators fitted on the calibration split only (Section 5)."""
    r0, k, y = calibration["R0"].to_numpy(), calibration["K"].to_numpy(), calibration[label].to_numpy()
    return dict(R_cal=Calibrator.fit(r0, y), K_cal=Calibrator.fit(k, y), Base=float(y.mean()),
                Platt_R=Calibrator.fit(r0, y, min_per_class_isotonic=len(y) + 1))


def predict_scores(maps: dict, items: pd.DataFrame) -> dict[str, np.ndarray]:
    r0, k = items["R0"].to_numpy(), items["K"].to_numpy()
    return dict(R_raw=r0, R_cal=maps["R_cal"].predict(r0), K_cal=maps["K_cal"].predict(k),
                Base=np.full(len(items), maps["Base"]), Platt_R=maps["Platt_R"].predict(r0))


def score_statistics(prefix: str, predictions: dict[str, np.ndarray], labels: np.ndarray, maps: dict,
                     notes: dict) -> dict[str, Statistic]:
    """ECE (both binnings), Brier, AUC and Brier skill over Base; notes get (distinct values, calibrator)."""
    base, statistics = WeightedScore(predictions["Base"], labels), {}
    for name, p in predictions.items():
        s, key = WeightedScore(p, labels), f"{prefix}:{name}|"
        notes[f"{prefix}:{name}"] = (len(np.unique(p)), getattr(maps.get(name), "method", "none"))
        statistics.update({key + "ece_equal_width": s.ece, key + "ece_equal_mass": lambda w, s=s: s.ece(w, True),
                           key + "brier": s.brier, key + "auc": s.auc,
                           key + "brier_skill": lambda w, s=s: 1 - s.brier(w) / base.brier(w)})
    return statistics


def curve_statistics(predictions: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, Statistic]:
    """Reliability-curve data; equal-mass bin edges fixed at the point estimate."""
    statistics = {}
    for name in CURVE_SCORES:
        s = WeightedScore(predictions[name], labels)
        statistics[f"curve:{name}|equal_width"] = lambda w, s=s: s.reliability(w, s.width_bins)
        statistics[f"curve:{name}|equal_mass"] = lambda w, s=s, m=s.mass_bins(np.ones(len(labels))): s.reliability(w, m)
    return statistics


def strata_statistics(test: pd.DataFrame, calibration: pd.DataFrame, predictions: np.ndarray) -> dict[str, Statistic]:
    """ECE, AUC, prevalence and size of R-cal in the ten strata of Section 8, and the worst stratum ECE."""
    cuts = np.quantile(calibration.drop_duplicates("scene_seed")["noise_px"], [1 / 3, 2 / 3])
    tercile, labels = np.digitize(test["noise_px"], cuts), test[LABEL].to_numpy()
    mismatch, other = test["mismatch"].to_numpy(), test["other_tuple_mismatch"].to_numpy()
    planar, tuples = test["planar_fraction"].to_numpy(), test["n_accepted_tuples"].to_numpy()
    masks = {"mismatch:p_mismatched": mismatch, "mismatch:other_tuple_point_mismatched": ~mismatch & other,
             "mismatch:clean": ~mismatch & ~other, **{f"sigma_px:tercile_{i + 1}": tercile == i for i in range(3)},
             "planar_fraction:<0.25": planar < 0.25, "planar_fraction:>=0.25": planar >= 0.25,
             "n_accepted_tuples:1": tuples == 1, "n_accepted_tuples:>1": tuples > 1}
    statistics, eces = {}, []
    for name, m in masks.items():
        s = WeightedScore(predictions[m], labels[m])
        eces.append(lambda w, s=s, m=m: s.ece(w[m]))
        statistics.update({f"strata:{name}|ece_equal_width": eces[-1], f"strata:{name}|auc": lambda w, s=s, m=m: s.auc(w[m]),
                           f"strata:{name}|prevalence": lambda w, m=m: float(w[m] @ labels[m] / w[m].sum()),
                           f"strata:{name}|n_items": lambda w, m=m: float(w[m].sum())})
    statistics["strata:worst|ece_equal_width"] = lambda w: max(f(w) for f in eces)
    return statistics


def comparison_rows(estimates: dict, draws: dict) -> list[dict]:
    """S1-S4 (intersection-union test), H1, H2 (Bonferroni over three) and ECE(R-cal) - ECE(Base)."""
    rows, p = [], f"{PRIMARY}:"

    def add(name: str, a: str, b: str | None, criterion: str, rule: Callable[..., bool] | None) -> None:
        values, estimate = draws[p + a] - (draws[p + b] if b else 0.0), estimates[p + a] - (estimates[p + b] if b else 0.0)
        (low, high), (low_b, high_b) = interval(values), interval(values, H2_LEVEL)
        rows.append(dict(comparison=name, estimate=float(estimate), ci_low=low, ci_high=high, ci_low_bonferroni=low_b,
                         ci_high_bonferroni=high_b, criterion=criterion,
                         passed=None if rule is None else bool(rule(low, high, low_b, high_b))))
    add("S1: ECE_ew(R-cal)", "R_cal|ece_equal_width", None, f"ci_high <= {S1_BOUND}", lambda lo, hi, *_: hi <= S1_BOUND)
    add("S2: ECE_ew(R-raw) - ECE_ew(R-cal)", "R_raw|ece_equal_width", "R_cal|ece_equal_width", "ci_low > 0",
        lambda lo, *_: lo > 0)
    add("S3: AUC(R-cal) - AUC(R-raw)", "R_cal|auc", "R_raw|auc", f"ci_low > {S3_MARGIN}", lambda lo, *_: lo > S3_MARGIN)
    add("S4: BSS(R-cal vs Base)", "R_cal|brier_skill", None, "ci_low > 0", lambda lo, *_: lo > 0)
    rows.append(dict(comparison="H1 success (S1-S4 all hold)", criterion="all of S1-S4",
                     passed=all(r["passed"] for r in rows)))
    for metric in ("auc", "brier", "ece_equal_width"):
        add(f"H2: {metric}(R-cal) - {metric}(K-cal)", f"R_cal|{metric}", f"K_cal|{metric}",
            "Bonferroni CI (98.33%) excludes 0", lambda lo, hi, lb, hb: lb > 0 or hb < 0)
    add("ECE_ew(R-cal) - ECE_ew(Base) (descriptive)", "R_cal|ece_equal_width", "Base|ece_equal_width", "descriptive", None)
    return rows


def summarise_counts(scene_counts: pd.DataFrame) -> pd.DataFrame:
    """Section 8 counts per split; Fisher tests of the exclusion rates against the calibration split."""
    c = scene_counts.assign(F_failed=scene_counts["excluded"] == "F_failed", with_items=scene_counts["items"] > 0,
                            few_inliers=scene_counts["excluded"] == "fewer_than_8_inliers",
                            n_included_points=(scene_counts["excluded"] == "") * N_POINTS)
    table = c.groupby("split", sort=False).agg(
        n_scenes=("scene_seed", "size"), excluded_F_failed=("F_failed", "sum"),
        excluded_fewer_than_8_inliers=("few_inliers", "sum"), scenes_with_items=("with_items", "sum"),
        ransac_surviving_points=("ransac_surviving", "sum"), n_included_points=("n_included_points", "sum"),
        surviving_mismatches=("surviving_mismatches", "sum"), points_outside_D=("outside_D", "sum"),
        items_excluded_true_undefined=("true_undefined_items", "sum"), items=("items", "sum"),
        single_tuple_items=("single_tuple_items", "sum"), accepted_tuples=("accepted_tuples", "sum"),
        tuples_sign_flip=("sign_flip", "sum"), tuples_reconstructed_undefined=("reconstructed_undefined", "sum"),
        tuples_true_undefined=("true_undefined", "sum")).reset_index()
    table["ransac_rejected_points"] = table.pop("n_included_points") - table["ransac_surviving_points"]
    table["share_outside_D"] = table["points_outside_D"] / table["ransac_surviving_points"].clip(lower=1)
    table["share_single_tuple_items"] = table.pop("single_tuple_items") / table["items"].clip(lower=1)
    excluded = {"scene": (table["excluded_F_failed"] + table["excluded_fewer_than_8_inliers"], table["n_scenes"]),
                "item": (table["points_outside_D"] + table["items_excluded_true_undefined"], table["ransac_surviving_points"])}
    for name, (bad, total) in excluded.items():
        table[f"fisher_p_{name}_exclusion_vs_calibration"] = [fisher_exact([[bad[0], total[0] - bad[0]], [b, t - b]])[1]
                                                               for b, t in zip(bad, total)]
        table[f"{name}_exclusion_rate_differs_flag"] = table[f"fisher_p_{name}_exclusion_vs_calibration"] < 0.05
    return table


def calibration_map_table(calibrator: Calibrator, threshold: float, r0_range: tuple[float, float]) -> pd.DataFrame:
    """Isotonic knots (or, for Platt, the logistic curve tabulated on the observed R0 range) and the population."""
    isotonic = calibrator.method == "isotonic"
    knots_x = calibrator.model.X_thresholds_ if isotonic else np.linspace(*r0_range, 201)
    knots_y = calibrator.model.y_thresholds_ if isotonic else calibrator.predict(knots_x)
    return pd.DataFrame(dict(x=knots_x, y=knots_y, input_convention=TRUST_INPUT_CONVENTION, conditioning_metric="kappa",
                             conditioning_threshold=threshold, label_tolerance=PRIMARY_TOLERANCE,
                             method=calibrator.method, domain=TRUST_DOMAIN))


def result_tables(split: str, estimates: dict, draws: dict, n_items: int, n_scenes: int,
                  notes: dict) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """Rows of trust_calibrated_metrics and trust_calibrated_reliability, and the trust_calibrated_strata table, from the statistic keys."""
    metrics, curves, strata = [], [], {}
    for key, estimate in estimates.items():
        (group, _, rest), (low, high) = key.partition(":"), interval(draws[key])
        name, metric = rest.split("|")
        if group == "curve":
            values, boot = np.reshape(estimate, (3, N_BINS)), draws[key].reshape(len(draws[key]), 3, N_BINS)
            curves += [dict(score=name, binning=metric, bin=b, mean_prediction=values[0, b], observed_frequency=values[1, b],
                            n_items=values[2, b], **dict(zip(("ci_low", "ci_high"), interval(boot[:, 1, b]))))
                       for b in range(N_BINS)]
        elif group == "strata":
            strata.setdefault(name, dict(stratum=name)).update({metric: float(estimate), f"{metric}_ci_low": low,
                                                                 f"{metric}_ci_high": high})
        else:
            metrics.append(dict(split=split, label="correspondence_correct" if group == "correspondence" else
                                f"eps_median<={group}", score=name, metric=metric, estimate=float(estimate),
                                ci_low=low, ci_high=high, boot_mean=float(np.nanmean(draws[key])),
                                boot_sd=float(np.nanstd(draws[key])), n_items=n_items, n_scenes=n_scenes,
                                **dict(zip(("n_distinct_predictions", "calibrator"),
                                           notes.get(f"{group}:{name}", (-1, "none"))))))
    table = pd.DataFrame(sorted(strata.values(), key=lambda row: row["stratum"]))
    if strata:
        named = table[table["stratum"] != "worst"]
        table["is_worst_stratum"] = table["stratum"] == named.loc[named["ece_equal_width"].idxmax(), "stratum"]
    return metrics, curves, table


def run_experiment(writer: Writer, seed_base: int = EXPERIMENT_SEED, splits: dict = SPLITS) -> pd.DataFrame:
    start, threshold = time.time(), calibrated_conditioning_threshold(metric="kappa")
    collected = {name: collect_split(name, offset, n, shifted, threshold, seed_base)
                 for name, (offset, n, shifted) in splits.items()}
    calibration, test, shifted = (collected[n][0] for n in ("calibration", "test", "shifted"))
    assert threshold == calibrated_conditioning_threshold(metric="kappa"), "threshold must be the calibrated kappa median"
    maps = fit_score_maps(calibration, LABEL)
    maps["R_desc_cal"] = Calibrator.fit(calibration["R_desc"].to_numpy(), calibration[LABEL].to_numpy())
    rng = np.random.default_rng(seed_base + BOOTSTRAP_SEED_OFFSET)
    (test_codes, n_test), (shifted_codes, n_shifted) = scene_codes(test), scene_codes(shifted)
    test_counts, shifted_counts = bootstrap_counts(rng, n_test), bootstrap_counts(rng, n_shifted)
    y_test, predictions, notes, shifted_notes = test[LABEL].to_numpy(), predict_scores(maps, test), {}, {}
    predictions.update(R_desc_raw=test["R_desc"].to_numpy(), R_desc_cal=maps["R_desc_cal"].predict(test["R_desc"]))
    statistics = score_statistics(PRIMARY, predictions, y_test, maps, notes)
    statistics.update({f"{PRIMARY}:K_raw|auc": WeightedScore(test["K"].to_numpy(), y_test).auc,
                       "correspondence:R_raw|auc": WeightedScore(test["R0"].to_numpy(), ~test["mismatch"]).auc,
                       **curve_statistics(predictions, y_test), **strata_statistics(test, calibration, predictions["R_cal"])})
    for tolerance in SECONDARY_TOLERANCES:
        tol_maps = fit_score_maps(calibration, f"Y_tol{tolerance}")
        tol_predictions = {k: v for k, v in predict_scores(tol_maps, test).items() if k in ("R_raw", "R_cal", "Base")}
        statistics.update(score_statistics(str(tolerance), tol_predictions, test[f"Y_tol{tolerance}"].to_numpy(),
                                           tol_maps, notes))
    estimates, draws = run_bootstrap(statistics, test_codes, test_counts)
    shifted_predictions = {k: v for k, v in predict_scores(maps, shifted).items() if k != "Platt_R"}
    shifted_estimates, shifted_draws = run_bootstrap(score_statistics(
        PRIMARY, shifted_predictions, shifted[LABEL].to_numpy(), maps, shifted_notes), shifted_codes, shifted_counts)
    print(f"  bootstraps done: {time.time() - start:.1f} s")
    metrics, curves, strata = result_tables("test", estimates, draws, len(test), n_test, notes)
    metrics += result_tables("shifted", shifted_estimates, shifted_draws, len(shifted), n_shifted, shifted_notes)[0]
    comparisons = pd.DataFrame(comparison_rows(estimates, draws))

    map_path = writer(calibration_map_table(maps["R_cal"], threshold, (calibration["R0"].min(), 1.0)),
                      "trust_calibrated_calibration_map.csv")
    shipped = load_trust_calibration(map_path, expected_threshold=threshold)
    assert np.allclose(shipped.predict(test["R0"].to_numpy()), predictions["R_cal"], atol=1e-12, rtol=0), \
        "the saved map must reproduce the fitted calibrator"
    writer(pd.concat([calibration, test, shifted], ignore_index=True), "trust_calibrated_items.csv")
    writer(pd.DataFrame(metrics), "trust_calibrated_metrics.csv")
    writer(comparisons, "trust_calibrated_comparisons.csv")
    writer(pd.DataFrame(curves), "trust_calibrated_reliability.csv")
    writer(strata, "trust_calibrated_strata.csv")
    writer(summarise_counts(pd.concat([collected[n][1] for n in splits], ignore_index=True)), "trust_calibrated_counts.csv")
    print(comparisons[["comparison", "estimate", "ci_low", "ci_high", "passed"]].to_string(index=False))
    worst = strata.loc[strata["is_worst_stratum"], "stratum"].iloc[0]
    print(f"  calibrator: {maps['R_cal'].method}; worst stratum: {worst}; runtime {time.time() - start:.1f} s")
    return comparisons


def save_with_provenance(table: pd.DataFrame, filename: str) -> Path:
    return save_results(table, filename, seed=EXPERIMENT_SEED, experiment_name=EXPERIMENT_NAME)


if __name__ == "__main__":
    run_experiment(save_with_provenance)
