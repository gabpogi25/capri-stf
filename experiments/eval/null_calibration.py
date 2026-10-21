"""
experiments/eval/null_calibration.py

Null calibration (docs/prereg/evaluation_plan.md).
Rigid (rigid) windows, n = 40, six views, at sigma in {0.25, 0.5, 1, 2} px, 600
windows each (indices < 300: calibration half, >= 300: evaluation half), plus 200
shifted-scene windows (60 Gaussian points, 20-degree arc, sigma = 0.5).

Per window: every scene statistic of mv_window (no legacy Delta_J), and for 5
random gated six-tuples (sigma_4 >= 0.02 in both Tyler-whitened reconstructions)
the Mahalanobis test of Delta_J = log|J(A)| - log|J(B)|:
  each subset's 6 points are re-triangulated linearly with that subset's BA cameras;
  Sigma = s^2 G G^T with G = d Delta_J / d(72 image coordinates) (central differences,
  cameras fixed, so camera uncertainty is NOT propagated); T = d^T Sigma^+ d,
  p = chi^2_3 survival; s = oracle sigma, sigma_hat (from the subset BA residuals),
  and 0.5 sigma (misspecified control). kappa = max kappa_J over A and B.
All decisions are made by e27_analysis_null.py from this output.
Seeds: 33,000,000 + k (k = window index in grid order). Output: null_calibration_windows.csv,
null_calibration_tuples.csv.
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
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import save_results  # noqa: E402
from algorithms.utils import whiten_homogeneous_points  # noqa: E402
from forensics.geometry.invariants import kappa_j, log_abs_invariants  # noqa: E402
from forensics.video.multiview import sample_gated_tuples  # noqa: E402
from synthetic_data.injection import arc_cameras, make_window  # noqa: E402
import mv_window as mw  # noqa: E402

SEED = 33_000_000 + (mw.SMOKE_SEED_OFFSET if mw.SMOKE else 0)
SIGMAS, PER_SIGMA, N_SHIFT, TUPLES_PER_WINDOW, FD_STEP = (0.25, 0.5, 1.0, 2.0), mw.shrink(600, 40), mw.shrink(200, 20), 5, 1e-7


def grid() -> list[tuple]:
    tasks = [(sigma, "rigid", i) for sigma in SIGMAS for i in range(PER_SIGMA)]
    tasks += [(0.5, "shift_gaussian", i) for i in range(N_SHIFT)]
    return [(*t, SEED + k) for k, t in enumerate(tasks)]


def triangulate(cameras: list[np.ndarray], views: list[np.ndarray]) -> np.ndarray:
    X = []
    for i in range(views[0].shape[0]):
        rows = [r for P, x in zip(cameras, views) for r in (x[i, 0] * P[2] - P[0], x[i, 1] * P[2] - P[1])]
        X.append(np.linalg.svd(np.array(rows))[2][-1])
    return np.array(X).T


def delta_j_map(u: np.ndarray, rec_a: dict, rec_b: dict, split: tuple) -> np.ndarray:
    pts = u.reshape(6, 6, 2)              # (view, point, xy)
    XA = triangulate(rec_a["cameras"], [pts[v] for v in split[0]])
    XB = triangulate(rec_b["cameras"], [pts[v] for v in split[1]])
    return log_abs_invariants(XA) - log_abs_invariants(XB)


def tuple_tests(views: list[np.ndarray], rec_a: dict, rec_b: dict, sigma_px: float, sigma_hat_px: float,
                rng: np.random.Generator) -> list[dict]:
    split = mw.SPLITS["interleaved"]
    XA, XB = whiten_homogeneous_points(rec_a["X"]), whiten_homogeneous_points(rec_b["X"])
    rows = []
    for t in sample_gated_tuples(XA, XB, TUPLES_PER_WINDOW, rng, mw.SIGMA4_MIN):
        try:
            u0 = np.stack([v[list(t)] for v in views]).ravel()
            d = delta_j_map(u0, rec_a, rec_b, split)
            G = np.stack([(delta_j_map(u0 + FD_STEP * e, rec_a, rec_b, split) - delta_j_map(u0 - FD_STEP * e, rec_a, rec_b, split))
                          / (2 * FD_STEP) for e in np.eye(len(u0))], axis=1)
            unit = lambda X: X / np.linalg.norm(X, axis=0)          # noqa: E731
            row = {"Tuple": "-".join(map(str, t)), "Kappa": max(kappa_j(unit(XA[:, list(t)])), kappa_j(unit(XB[:, list(t)])))}
            w, V = np.linalg.eigh(G @ G.T)
            keep = w > 1e-12 * w.max()
            quad = float(np.sum((V[:, keep].T @ d) ** 2 / w[keep]))
            for name, s in (("oracle", sigma_px), ("hat", sigma_hat_px), ("half", 0.5 * sigma_px)):
                T = quad / (s / mw.HALF) ** 2
                row[f"T_{name}"], row[f"p_{name}"] = T, float(stats.chi2.sf(T, int(keep.sum())))
            row["Dof"] = int(keep.sum())
            rows.append(row)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:   # logged, never dropped silently
            rows.append({"Tuple": "-".join(map(str, t)), "Error": type(error).__name__})
    return rows


def run(task: tuple) -> tuple[dict, list[dict]]:
    sigma, kind, index, seed = task
    rng = np.random.default_rng(seed)
    if kind == "rigid":
        window = make_window(rng, "rigid", 0.0, sigma)
    else:
        world = np.clip(rng.normal(0.0, 0.6, (60, 3)), -1.5, 1.5)
        window = make_window(rng, "rigid", 0.0, sigma, cameras=arc_cameras(6, arc_deg=20.0), world=world)
    row = mw.window_statistics(window["tracks"], sigma, seed, with_legacy=False)
    recs = row.pop("_rec", None)
    meta = {"Sigma": sigma, "Kind": kind, "Index": index, "Half": "calibration" if index < PER_SIGMA // 2 else "evaluation",
            "Seed": seed}
    tuples = []
    if recs is not None:
        views = [mw.normalize(t) for t in window["tracks"]]
        try:
            tuples = [{**meta, **r} for r in tuple_tests(views, *recs, sigma, row["sigma_hat_px"], rng)]
        except (np.linalg.LinAlgError, ValueError) as error:          # e.g. whitening or gating failed
            tuples = [{**meta, "Tuple": "", "Error": f"window: {type(error).__name__}"}]
    return {**meta, **row}, tuples


def main() -> None:
    with ProcessPoolExecutor(4) as pool:
        results = list(pool.map(run, grid(), chunksize=8))
    windows = pd.DataFrame([w for w, _ in results])
    tuples = pd.DataFrame([t for _, ts in results for t in ts])
    print(windows.groupby(["Kind", "Sigma"])["abstain"].mean())
    save_results(windows, mw.out_name("null_calibration_windows.csv"), SEED, "null calibration: rigid windows")
    save_results(tuples, mw.out_name("null_calibration_tuples.csv"), SEED, "null calibration: tuple-level Mahalanobis Delta_J tests")


if __name__ == "__main__":
    main()
