"""
experiments/eval/consistency_test_windows.py

Windows for docs/prereg/consistency_test.md (pre-registered; decision rules in
consistency_test_analysis.py). Every window gets every statistic of mv_window.window_statistics
(pencil_q10, acat, split_glrt with the generating sigma, glrt_all, f_ratio, sigma_hat_px,
sampson_mean, ...) plus:
  std_native_q10   Q_0.1 of D_t / (|g_t(A)| + |g_t(B)|), each reconstruction in its own frame
                   (the standardisation as evaluated in mv_window; 20,000 sampled pencils)
  std_tyler_q10    the same after Tyler whitening of each reconstruction (common canonical frame)
  frame_q50        median of the 3(n - 5) frame-coordinate discrepancies (algorithms/consistency/frame.py)

Grid (n = 40 points, six views, interleaved split; scene S0 = uniform cube, 30-degree arc):
  cal       S0, rigid, sigma 0.5 px, 400 windows (thresholds and conformal p-values)
  test      S0, rigid, sigma 0.5 px, 400 windows (size)
  sig025    S0, rigid, sigma 0.25 px, 200 windows; sig1: S0, rigid, sigma 1.0 px, 200 windows
  shift     S1 = 60 Gaussian points, 20-degree arc, rigid, sigma 0.5 px, 200 windows
  jitter 2 px, drift 4 px, warp 2 px, swap 0.05: S0, sigma 0.5 px, 400 windows each
Seeds: window k uses 81,000,000 + k in the task order below (smoke runs: +60,000,000).
Output: consistency_test_windows.csv.
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
from algorithms.consistency import variants  # noqa: E402
from algorithms.consistency.frame import frame_discrepancies  # noqa: E402
from algorithms.utils import whiten_homogeneous_points  # noqa: E402
from synthetic_data.injection import arc_cameras, make_window  # noqa: E402
import mv_window as mw  # noqa: E402

SEED = 81_000_000 + (mw.SMOKE_SEED_OFFSET if mw.SMOKE else 0)
CELLS = [("cal", "rigid", 0.0, 0.5, "S0", 400), ("test", "rigid", 0.0, 0.5, "S0", 400),
         ("sig025", "rigid", 0.0, 0.25, "S0", 200), ("sig1", "rigid", 0.0, 1.0, "S0", 200),
         ("shift", "rigid", 0.0, 0.5, "S1", 200),
         ("jitter", "jitter", 2.0, 0.5, "S0", 400), ("drift", "drift", 4.0, 0.5, "S0", 400),
         ("warp", "warp", 2.0, 0.5, "S0", 400), ("swap", "swap", 0.05, 0.5, "S0", 400)]
NEW_KEYS = ("std_native_q10", "std_tyler_q10", "frame_q50")


def tasks() -> list[tuple]:
    out = []
    for role, family, dose, sigma, scene, count in CELLS:
        out += [(role, family, dose, sigma, scene) for _ in range(mw.shrink(count, 8))]
    return [t + (SEED + k,) for k, t in enumerate(out)]


def extra_statistics(X_a: np.ndarray, X_b: np.ndarray, seed: int) -> dict:
    n = X_a.shape[1]
    pencils = variants.sampled_pencils(n, mw.N_PENCILS_SMALL, seed + 1)
    out = {}
    for key, (A, B) in (("std_native_q10", (X_a, X_b)),
                        ("std_tyler_q10", (whiten_homogeneous_points(X_a), whiten_homogeneous_points(X_b)))):
        z = variants.standardized_discrepancies(A, B, pencils, 1.0)
        z = z[np.isfinite(z)]
        out[key] = float(np.quantile(z, 0.1)) if z.size else float("nan")
    D = frame_discrepancies(X_a, X_b, seed=seed)
    D = D[np.isfinite(D)]
    out["frame_q50"] = float(np.median(D)) if D.size else float("nan")
    return out


def run_window(task: tuple) -> dict:
    role, family, dose, sigma, scene, seed = task
    rng = np.random.default_rng(seed)
    if scene == "S1":
        world = rng.normal(0.0, 0.5, size=(60, 3))
        window = make_window(rng, family, dose, sigma, cameras=arc_cameras(6, arc_deg=20.0), world=world)
    else:
        window = make_window(rng, family, dose, sigma)
    row = mw.window_statistics(window["tracks"], sigma, seed, with_legacy=False, moved=window["moved"])
    rec = row.pop("_rec", None)
    if not row["abstain"] and rec is not None:
        try:
            row.update(extra_statistics(rec[0]["X"], rec[1]["X"], seed))
        except mw.GUARDED as error:
            row.update({k: float("nan") for k in NEW_KEYS})
            row["errors"] = (row.get("errors", "") + f"; extra: {type(error).__name__}").strip("; ")
    return {"Role": role, "Family": family, "Dose": dose, "Sigma": sigma, "Scene": scene, "Seed": seed,
            "Induced_px": window.get("induced_px", np.nan), **row}


def main() -> None:
    work = tasks()
    print(f"consistency_test: {len(work)} windows (smoke: {mw.SMOKE})")
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(run_window, work, chunksize=4))
    df = pd.DataFrame(rows)
    save_results(df, mw.out_name("consistency_test_windows.csv"), SEED,
                 "consistency_test windows (docs/prereg/consistency_test.md)")


if __name__ == "__main__":
    main()
