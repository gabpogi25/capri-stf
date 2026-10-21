"""
experiments/eval/real_pencil_size_ci.py

Scene-clustered 95% intervals for the false-alarm rates of docs/prereg/real_pencil.md
(descriptive; not part of the pre-registered decisions, which stand as recorded in
real_pencil_decisions.csv).

Windows of one scene are correlated, so the interval resamples test scenes with replacement
(percentile bootstrap, 2,000 resamples). The calibration scenes, and therefore the thresholds,
are held fixed, so the intervals do not include calibration uncertainty. Per-scene sizes are
written as well.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "experiments" / "eval")]

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402
from real_pencil_analysis import STATS, Calibration  # noqa: E402

SEED, N_BOOTSTRAP = 87_600_000, 2_000


def main() -> None:
    df = pd.concat([pd.read_csv(RESULTS_DIR / f"real_pencil_windows_batch{b}.csv", comment="#")
                    for b in (1, 2)], ignore_index=True)
    df = df.assign(Unit=df["Dataset"] + "/" + df["Scene"])
    null = df[df["Injection"] == "none"]
    calib = Calibration(null[null["Role"] == "cal"])
    test_null = null[null["Role"] == "test"]
    scenes = np.array(sorted(test_null["Unit"].unique()))
    reject = {s: pd.Series(calib.reject(test_null, s), index=test_null.index) for s in STATS}
    per_scene = {s: [(int(reject[s][test_null["Unit"] == u].sum()), int((test_null["Unit"] == u).sum()))
                     for u in scenes] for s in STATS}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(scenes), size=(N_BOOTSTRAP, len(scenes)))
    rows, scene_rows = [], []
    for s in STATS:
        k = np.array([a for a, _ in per_scene[s]]); n = np.array([b for _, b in per_scene[s]])
        boot = k[draws].sum(axis=1) / n[draws].sum(axis=1)
        low, high = np.percentile(boot, [2.5, 97.5])
        rows.append({"Statistic": s, "Size": k.sum() / n.sum(), "CI_Low": low, "CI_High": high,
                     "N_Test_Windows": int(n.sum()), "N_Test_Scenes": int((n > 0).sum())})
        for u, (a, b) in zip(scenes, per_scene[s]):
            scene_rows.append({"Statistic": s, "Scene": u, "Rejections": a, "Windows": b, "Size": a / b})
    save_results(pd.DataFrame(rows), "real_pencil_size_ci.csv", SEED,
                 "Real-data pencil test: scene-clustered size intervals (descriptive)")
    save_results(pd.DataFrame(scene_rows), "real_pencil_size_by_scene.csv", SEED,
                 "Real-data pencil test: size per test scene (descriptive)")


if __name__ == "__main__":
    main()
