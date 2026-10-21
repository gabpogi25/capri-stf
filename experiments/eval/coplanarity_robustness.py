"""
experiments/eval/coplanarity_robustness.py

representation robustness check (docs/prereg/evaluation_plan.md): does the coplanarity score's win
(matched degeneracy experiment, H1: capri_coplanarity AUC 0.983 vs DEGENSAC inlier count 0.937) survive
other representations and base triples? Fresh seeds, same generator
(run_synthetic_baselines.make_scene), same M8 arm (8 correspondences only), same
label (>= 6 of 8 points planar), same sample design (12 samples per k = 0..8).

capri_coplanarity variants (m = 6 stratum distance, degeneracy pre-registration Section 2, NaN -> lowest):
  R0 unit columns, largest-triangle base triple (the pre-registered score; replication)
  R1 affine representatives X / X_4
  R2 fixed PGL(4) frame A = N(0, 1)^{4x4} + 2 I per sample, then unit columns
  R3 base triple = first three points of each subset
  R4 base triple = smallest view-1 triangle
  R5 base triple = random (seeded per sample)
New arm tyler_sigma6: -log10 min over 6-subsets of sigma_4 of the unit columns of the
Tyler-whitened 8-point reconstruction. Tyler whitening is equivariant and per-point
scale free, so this score does not depend on the frame or the representatives
(checked in tests/test_eval_variants.py). It is not noise-whitened.
Baselines: degensac_h_inliers, degensac_e6, homography_ls6, gric_h_minus_f (matched degeneracy experiment code).
Seeds: trial t < 80 uses 35,000,000 + t; bootstrap 35,900,000. Output:
coplanarity_robustness_samples.csv, coplanarity_robustness_comparisons.csv.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import save_results  # noqa: E402
import run_degeneracy_matched as x14  # noqa: E402
import run_synthetic_baselines as rsb  # noqa: E402
from algorithms.degeneracy import coplanarity as cop  # noqa: E402
from algorithms.utils import whiten_homogeneous_points  # noqa: E402

SEED, BOOT_SEED, N_TRIALS, N_BOOT = 35_000_000, 35_900_000, 80, 2000
VARIANTS = ("R0", "R1", "R2", "R3", "R4", "R5")


def base_triples(rule: str, view_1: np.ndarray, subsets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rule == "largest":
        return cop.largest_triangle_base_triples(view_1, subsets)
    if rule == "first":
        return subsets[:, :3]
    triples = subsets[:, cop.subsets_of_size(subsets.shape[1], 3)]
    if rule == "random":
        return triples[np.arange(len(subsets)), rng.integers(0, triples.shape[1], len(subsets))]
    corners = np.concatenate([view_1[triples], np.ones(triples.shape + (1,))], axis=3)
    return triples[np.arange(len(subsets)), np.argmin(np.abs(np.linalg.det(corners)), axis=1)]


def tyler_sigma6(points: np.ndarray) -> float:
    if not np.all(np.isfinite(points)):
        return float("nan")
    W = whiten_homogeneous_points(points)
    W = W / np.linalg.norm(W, axis=0)
    mats = np.moveaxis(W[:, cop.subsets_of_size(W.shape[1], 6)], 0, 1)
    return float(-np.log10(np.min(np.linalg.svd(mats, compute_uv=False)[:, 3])))


def score_sample(view_1: np.ndarray, view_2: np.ndarray, rng: np.random.Generator) -> dict:
    chart_1, chart_2 = rsb.hartley_normalize(view_1)[1], rsb.hartley_normalize(view_2)[1]
    perturbed = cop.batch_eight_point_reconstruction(*cop.perturbed_correspondences(view_1, view_2, x14.STEP_PX),
                                                     chart_1, chart_2)[0]
    subsets = cop.subsets_of_size(8, 6)
    frame = rng.standard_normal((4, 4)) + 2 * np.eye(4)
    representations = {"R0": (perturbed, "largest"), "R1": (perturbed / perturbed[:, 3:4, :], "largest"),
                       "R2": (np.einsum("ij,bjn->bin", frame, perturbed), "largest"),
                       "R3": (perturbed, "first"), "R4": (perturbed, "smallest"), "R5": (perturbed, "random")}
    row = {}
    for name, (points, rule) in representations.items():
        if name == "R2":
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        constraints = cop.bracket_constraints(points, subsets, base_triples(rule, view_1, subsets, rng))
        distances, status = cop.sampson_distances(constraints, x14.STEP_PX, rsb.NOISE_PX)
        row[f"cop_{name}"] = cop.coplanarity_score(distances, status)
    row["tyler_sigma6"] = tyler_sigma6(perturbed[0])
    errors_4, errors_6 = (x14.homography_subset_errors(view_1, view_2, x14.SUBSETS_4),
                          x14.homography_subset_errors(view_1, view_2, x14.SUBSETS_6))
    row.update({"degensac_h_inliers": rsb.score_degensac(view_1, view_2), "gric_h_minus_f": rsb.score_gric(view_1, view_2),
                "degensac_e6": x14.sixth_smallest_error_score(errors_4),
                "homography_ls6": x14.subset_rms_error_score(errors_6, x14.SUBSETS_6)})
    return row


def run_trial(trial: int) -> list[dict]:
    import cv2
    cv2.setNumThreads(1)
    seed = SEED + trial
    rng = np.random.default_rng(seed)
    view_1, view_2, planar = rsb.make_scene(rng)
    planar_idx, generic_idx = np.flatnonzero(planar), np.flatnonzero(~planar)
    rows = []
    for k in range(9):
        for _ in range(x14.SAMPLES_PER_K):
            sample = np.r_[rng.choice(planar_idx, k, replace=False), rng.choice(generic_idx, 8 - k, replace=False)]
            rows.append({"Trial": trial, "Seed": seed, "N_Planar": k, "Degenerate": int(k >= rsb.DEGENERATE_MIN_PLANAR),
                         **score_sample(view_1[sample], view_2[sample], rng)})
    return rows


def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    y = df["Degenerate"].to_numpy()
    scores = {c: x14.map_nonfinite(df[c].to_numpy(float), "primary") for c in df.columns
              if c.startswith("cop_") or c in ("tyler_sigma6", "degensac_h_inliers", "gric_h_minus_f", "degensac_e6", "homography_ls6")}
    trials = df["Trial"].to_numpy()
    groups = [np.flatnonzero(trials == t) for t in np.unique(trials)]
    boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]
    plan = [(f"H1_{v}", f"cop_{v}", "degensac_h_inliers", 0.99 if v != "R0" else 0.95) for v in VARIANTS]
    plan += [("H_T", "tyler_sigma6", "degensac_h_inliers", 0.95), ("TOST_T", "tyler_sigma6", "homography_ls6", 0.90)]
    rows = []
    for name, a, b, level in plan:
        diff = auc(y, scores[a]) - auc(y, scores[b])
        draws = np.array([auc(y[i], scores[a][i]) - auc(y[i], scores[b][i]) for i in boots])
        lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
        verdict = (("equivalent" if lo > -0.01 and hi < 0.01 else "not shown equivalent") if name.startswith("TOST")
                   else ("first better" if lo > 0 else "second better" if hi < 0 else "no difference shown"))
        rows.append({"Hypothesis": name, "A": a, "B": b, "AUC_A": auc(y, scores[a]), "AUC_B": auc(y, scores[b]),
                     "Difference": diff, "CI_Level": level, "CI_Lo": lo, "CI_Hi": hi, "Verdict": verdict})
    out = pd.DataFrame(rows)
    survives = bool((out[out.Hypothesis.str.startswith("H1_")]["CI_Lo"] > 0).all())
    out["A_claim_survives_all_representations"] = survives
    return out


def main() -> None:
    with ProcessPoolExecutor(4) as pool:
        df = pd.DataFrame([r for rows in pool.map(run_trial, range(N_TRIALS)) for r in rows])
    out = comparisons(df)
    print(out.to_string(index=False))
    save_results(df, "coplanarity_robustness_samples.csv", SEED, "representation robustness check: coplanarity score under other representations (fresh seeds)")
    save_results(out, "coplanarity_robustness_comparisons.csv", BOOT_SEED, "representation robustness check: pre-registered comparisons")


if __name__ == "__main__":
    main()
