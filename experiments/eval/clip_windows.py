"""
experiments/eval/clip_windows.py

clip aggregation (docs/prereg/evaluation_plan.md): clip-level data.
A clip is one rigid point set (n = 40) seen by 48 cameras on a 240-degree arc, cut
into 8 windows of 6 consecutive views (30 degrees each). Inconsistency is injected
independently in every window (jitter/drift/warp at the family's dose d*, fixed by the rule in
evaluation_plan from injection_windows.csv), or from window 4 on (change clips, drift at d*).

Roles: calibration (400 single rigid windows at random arc positions), rigid (300 null clips),
jitter/drift/warp (100 clips each), change (100 clips). Per window: every scene statistic of
mv_window (no legacy Delta_J) and pencil_q10 at K in {100, 300, 1000, 3000, 10000,
50000} sampled pencils. Extras: the contiguous split {0,1,2}/{3,4,5} for null clips
0-49 and family clips 0-29; the U-statistic (mean pencil_q10 over the 10 balanced
3/3 view partitions) for window 0 of null and drift clips 0-99.
Analysis: clip_analysis.py. Seeds 34,000,000 + task index. Output:
clip_windows.csv.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402
from algorithms.consistency.pencil import pencil_discrepancies  # noqa: E402
from synthetic_data.injection import arc_cameras, make_window, scene_points  # noqa: E402
import mv_window as mw  # noqa: E402

SEED = 34_000_000 + (mw.SMOKE_SEED_OFFSET if mw.SMOKE else 0)
N_WINDOWS, NOISE = 8, 0.5
K_SWEEP = (100, 300, 1000, 3000, 10_000, 50_000)
PARTITIONS = [(a, tuple(v for v in range(6) if v not in a)) for a in ((0,) + c for c in combinations(range(1, 6), 2))]


def detectable_doses() -> dict[str, float]:
    """d* per family: smallest injection grid dose with pencil_q10 power >= 0.5 at sigma = 0.5 (threshold = 95th
    percentile of the rigid calibration windows); 16 px if none (evaluation_plan, clip aggregation)."""
    df = pd.read_csv(RESULTS_DIR / mw.out_name("injection_windows.csv"), comment="#")
    df = df[(df.Sigma == 0.5) & ~df.abstain.astype(bool)]
    threshold = df[(df.Family == "rigid") & (df.Role == "calibration")]["pencil_q10"].quantile(0.95)
    out = {}
    for family in ("jitter", "drift", "warp"):
        power = df[df.Family == family].groupby("Dose")["pencil_q10"].apply(lambda s: np.mean(s > threshold))
        hits = power[power >= 0.5]
        out[family] = float(hits.index.min()) if len(hits) else 16.0
    return out


def tasks(doses: dict[str, float]) -> list[tuple]:
    out = [("calibration", i, i % N_WINDOWS, "rigid", 0.0) for i in range(mw.shrink(400, 40))]
    clips = [("rigid", c) for c in range(mw.shrink(300, 8))] + [(f, c) for f in ("jitter", "drift", "warp") for c in range(mw.shrink(100, 4))]
    clips += [("change", c) for c in range(mw.shrink(100, 4))]
    for role, clip in clips:
        for w in range(N_WINDOWS):
            family = role if role in ("jitter", "drift", "warp") else ("drift" if role == "change" and w >= 4 else "rigid")
            out.append((role, clip, w, family, doses.get(family, 0.0) if family != "rigid" else 0.0))
    return [(*t, SEED + k) for k, t in enumerate(out)]


def clip_world(role: str, clip: int) -> np.ndarray:
    """The clip's point set depends only on (role, clip), so all its windows share it."""
    offset = {"calibration": 0, "rigid": 1, "jitter": 2, "drift": 3, "warp": 4, "change": 5}[role]
    return scene_points(np.random.default_rng(SEED + 9_000_000 + 1000 * offset + clip), 40)


def u_statistic(views: list[np.ndarray], seed: int) -> float:
    values = []
    for a, b in PARTITIONS:
        try:
            rec_a = mw.reconstruct([views[v] for v in a], seed)
            rec_b = mw.reconstruct([views[v] for v in b], seed)
        except Exception:                      # noqa: BLE001  a failed partition is left out and counted
            continue
        D = pencil_discrepancies(rec_a["X"], rec_b["X"], seed=seed)
        values.append(np.quantile(D[np.isfinite(D)], 0.1))
    return float(np.mean(values)) if values else float("nan")


def run(task: tuple) -> dict:
    role, clip, w, family, dose, seed = task
    world = (scene_points(np.random.default_rng(SEED + 9_500_000 + clip), 40) if role == "calibration"
             else clip_world(role, clip))
    cameras = arc_cameras(6, arc_deg=30.0, start_deg=-120.0 + 30.0 * w)
    window = make_window(np.random.default_rng(seed), family, dose, NOISE, cameras=cameras, world=world)
    row = mw.window_statistics(window["tracks"], NOISE, seed, with_legacy=False)
    recs = row.pop("_rec", None)
    meta = {"Role": role, "Clip": clip, "Window": w, "Family": family, "Dose": dose, "Seed": seed}
    if recs is None:
        return {**meta, **row}
    for K in K_SWEEP:
        D = pencil_discrepancies(recs[0]["X"], recs[1]["X"], max_pencils=K, seed=seed)
        row[f"pencil_q10_K{K}"] = float(np.quantile(D[np.isfinite(D)], 0.1))
    if (role == "rigid" and clip < mw.shrink(50, 4)) or (role in ("jitter", "drift", "warp") and clip < mw.shrink(30, 2)):
        contiguous = mw.window_statistics(window["tracks"], NOISE, seed, split="contiguous", with_legacy=False)
        row["pencil_q10_contiguous"] = contiguous.get("pencil_q10", float("nan"))
    if w == 0 and role in ("rigid", "drift") and clip < mw.shrink(100, 2):
        row["pencil_q10_U"] = u_statistic([mw.normalize(t) for t in window["tracks"]], seed)
    return {**meta, **row}


def main() -> None:
    doses = detectable_doses()
    print("d* per family:", doses)
    with ProcessPoolExecutor(4) as pool:
        df = pd.DataFrame(list(pool.map(run, tasks(doses), chunksize=8)))
    df["d_star_jitter"], df["d_star_drift"], df["d_star_warp"] = doses["jitter"], doses["drift"], doses["warp"]
    save_results(df, mw.out_name("clip_windows.csv"), SEED, "clip aggregation: clip windows (aggregation, K-sweep, split, U-statistic)")


if __name__ == "__main__":
    main()
