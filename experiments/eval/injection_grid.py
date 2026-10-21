"""
experiments/eval/injection_grid.py

injection grid (docs/prereg/evaluation_plan.md): point-level
injection harness. Every window of the grid gets every statistic of
mv_window.window_statistics; the analysis (kill switch, dose-response, size,
power, the detector benchmark) is in detector_benchmark.py and reads only this output.

Grid (n = 40 points, 6 views, interleaved split):
  sigma = 0.5 px:  rigid calibration 400, rigid test 400; global_projective/jitter/drift/warp at doses
                   {0.5, 1, 2, 4, 8, 16} px x 100; swap at swap fractions
                   {0.05, 0.1, 0.2, 0.3} x 100; planar x 100.
  sigma = 0.25, 1.0 px: rigid calibration 200, rigid test 200; jitter/drift/warp at 8 px x 100.
Deterministic checks: (a) noise-free rigid and global_projective (8 px) windows, 20 each: pencil_q10
must be < 1e-6 (global_projective is consistent with a rigid scene); (b) Jacobian rank of the full
pencil family (n = 12) with respect to the 72 image coordinates of one subset
(true cameras, linear triangulation) is 3n - 15 = 21: with central differences at
h = 1e-6 and 1e-7, s_21 changes by < 1% and s_22 falls by a factor >= 30 (it is
finite-difference error, O(h^2)); required on >= 95% of 200 configurations. (A fixed
gap s_21 / s_22 >= 1e6 is not attainable: s_21 / s_1 is about 3e-4 on exploration
seeds.)
Seeds: window k uses 32,000,000 + k (grid order); checks 32,900,000 + i.
Output: injection_windows.csv, injection_checks.csv.
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
from algorithms.consistency.pencil import log_abs_pencil_cross_ratios, pencil_drift_score  # noqa: E402
from synthetic_data.injection import arc_cameras, make_window, project  # noqa: E402
import mv_window as mw  # noqa: E402

SEED, CHECK_SEED = 32_000_000 + (mw.SMOKE_SEED_OFFSET if mw.SMOKE else 0), 32_900_000 + (mw.SMOKE_SEED_OFFSET if mw.SMOKE else 0)
DOSES, SWAPS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0), (0.05, 0.1, 0.2, 0.3)


def grid() -> list[tuple]:
    cells = [(0.5, "rigid", 0.0, "calibration", 400), (0.5, "rigid", 0.0, "test", 400)]
    cells += [(0.5, f, d, "test", 100) for f in ("global_projective", "jitter", "drift", "warp") for d in DOSES]
    cells += [(0.5, "swap", d, "test", 100) for d in SWAPS] + [(0.5, "planar", 0.0, "test", 100)]
    for sigma in (0.25, 1.0):
        cells += [(sigma, "rigid", 0.0, "calibration", 200), (sigma, "rigid", 0.0, "test", 200)]
        cells += [(sigma, f, 8.0, "test", 100) for f in ("jitter", "drift", "warp")]
    tasks, k = [], 0
    for sigma, family, dose, role, count in cells:
        for i in range(mw.shrink(count, 12 if family == "rigid" else 6)):
            tasks.append((sigma, family, dose, role, i, SEED + k))
            k += 1
    return tasks


def run(task: tuple) -> dict:
    sigma, family, dose, role, index, seed = task
    window = make_window(np.random.default_rng(seed), family, dose, sigma)
    row = mw.window_statistics(window["tracks"], sigma, seed, moved=window["moved"])
    row.pop("_rec", None)
    return {"Sigma": sigma, "Family": family, "Dose": dose, "Role": role, "Index": index, "Seed": seed,
            "Induced_px": window["induced_px"], "N_moved": int(window["moved"].sum()), **row}


def noise_free_checks() -> list[dict]:
    rows = []
    for family in ("rigid", "global_projective"):
        for i in range(mw.shrink(20, 2)):
            seed = CHECK_SEED + (0 if family == "rigid" else 100) + i
            window = make_window(np.random.default_rng(seed), family, 8.0 if family == "global_projective" else 0.0, 0.0)
            row = mw.window_statistics(window["tracks"], 1e-6, seed, with_legacy=False)
            rows.append({"Check": f"noise_free_{family}", "Index": i, "Value": row.get("pencil_q10", np.nan),
                         "Pass": bool(row.get("pencil_q10", np.inf) < 1e-6)})
    return rows


def rank_check(index: int) -> dict:
    rng = np.random.default_rng(CHECK_SEED + 1000 + index)
    cameras = [mw_normalizer() @ P for P in arc_cameras(6)[0::2]]
    world = rng.uniform(-1, 1, (12, 3))
    u0 = np.concatenate([(np.c_[project(P, world), np.ones(12)] @ mw_normalizer().T)[:, :2].ravel()
                         for P in arc_cameras(6)[0::2]])

    def family(u):
        pts = u.reshape(3, 12, 2)
        X = []
        for i in range(12):
            rows = [r for P, x in zip(cameras, pts) for r in (x[i, 0] * P[2] - P[0], x[i, 1] * P[2] - P[1])]
            X.append(np.linalg.svd(np.array(rows))[2][-1])
        return log_abs_pencil_cross_ratios(np.array(X).T)

    spectra = {}
    for h in (1e-6, 1e-7):
        J = np.stack([(family(u0 + h * e) - family(u0 - h * e)) / (2 * h) for e in np.eye(len(u0))], axis=1)
        spectra[h] = np.linalg.svd(J[np.all(np.isfinite(J), axis=1)], compute_uv=False)
    s6, s7 = spectra[1e-6], spectra[1e-7]
    stable_21 = abs(s7[20] / s6[20] - 1.0) < 1e-2          # a true singular value does not depend on h
    artefact_22 = s6[21] / max(s7[21], 1e-300) >= 30.0      # finite-difference error falls like h^2
    return {"Check": "rank_3n_minus_15", "Index": index, "Value": float(s7[20] / s7[0]),
            "S22_ratio_h6_h7": float(s6[21] / max(s7[21], 1e-300)), "Pass": bool(stable_21 and artefact_22)}


def mw_normalizer() -> np.ndarray:
    return np.array([[1 / mw.HALF, 0, -mw.CENTRE[0] / mw.HALF], [0, 1 / mw.HALF, -mw.CENTRE[1] / mw.HALF], [0, 0, 1.0]])


def main() -> None:
    tasks = grid()
    print(f"{len(tasks)} windows")
    with ProcessPoolExecutor(4) as pool:
        windows = pd.DataFrame(list(pool.map(run, tasks, chunksize=8)))
        ranks = list(pool.map(rank_check, range(mw.shrink(200, 4))))
    checks = pd.DataFrame(noise_free_checks() + ranks)
    print(checks.groupby("Check")["Pass"].agg(["mean", "count"]))
    save_results(windows, mw.out_name("injection_windows.csv"), SEED, "injection grid: every statistic per window")
    save_results(checks, mw.out_name("injection_checks.csv"), CHECK_SEED, "injection grid: noise-free and Jacobian-rank checks")


if __name__ == "__main__":
    main()
