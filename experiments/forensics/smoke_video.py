"""
experiments/forensics/smoke_video.py

Exploration (not evidence): time the video statistics and count abstentions on a few items of
a manifest before the forensic pre-registration fixes the batch sizes.

Draws --per_class real and --per_class generated items at random (exploration seed), runs
media_statistics.video_table on them with an exploration seed, and writes
smoke_video_statistics.csv (save_results) with the seconds per item, the number of usable
pencil windows and of usable frames per item, and whether the item abstained.

Usage:
  python experiments/forensics/smoke_video.py --manifest data/manifests/videos.csv --per_class 10
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]

from experiment_utils import save_results  # noqa: E402
from forensics.config import load_config  # noqa: E402
from forensics.io.manifests import load_manifest  # noqa: E402
from media_statistics import video_table  # noqa: E402

SMOKE_SEED = 88_900_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per_class", type=int, default=10)
    parser.add_argument("--splits", nargs="+", default=["val", "cal"],
                        help="draw only from these splits (never test while settings are explored)")
    args = parser.parse_args()
    manifest = load_manifest(str(args.manifest))
    assert "test" not in args.splits, "exploration never touches the test split"
    manifest = manifest[manifest["split"].isin(args.splits)]
    rng = np.random.default_rng(SMOKE_SEED)
    picks = [manifest[manifest["label"] == label].sample(args.per_class, random_state=rng.integers(2 ** 31))
             for label in ("real", "generated")]
    config = load_config()
    rows = []
    for k, item in pd.concat(picks).reset_index(drop=True).iterrows():
        t0 = time.perf_counter()
        table = video_table(item.to_frame().T, config, SMOKE_SEED + 1000 * k)
        seconds = time.perf_counter() - t0
        row = table.iloc[0]
        rows.append({"path": item["path"], "label": item["label"], "generator": item["generator"],
                     "seconds": seconds, "n_windows": len(json.loads(row["window_statistics"])),
                     "n_frames": len(json.loads(row["frame_statistics"])), "abstain": bool(row["abstain"])})
        print(json.dumps(rows[-1]), flush=True)
    save_results(pd.DataFrame(rows), "smoke_video_statistics.csv", SMOKE_SEED,
                 "Exploration: video statistics timing and abstention on a few manifest items")


if __name__ == "__main__":
    main()
