"""
experiments/forensics/video_fixture.py

End-to-end check of the video branch on rendered mp4 clips (textured room,
moving camera). Inconsistent clips add a radial warp whose coefficient
oscillates over time, so the frames are not views of one rigid scene.
Reports the per-window pencil drift statistic through the full path: decode ->
KLT tracking -> rigid segmentation -> parallax gate -> two subset
reconstructions -> pencil drift statistic. A functional test of the code;
it is not evidence about generated video.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import save_csv
from forensics.config import load_config
from forensics.pipeline_video import window_statistic
from forensics.video.decode import decode
from forensics.video.synthetic import render_synthetic_video

SEED = 42207
WARPS = (0.0, 0.03, 0.08)
N_CLIPS = 3


def main() -> None:
    config = load_config(overrides={"video": {"view_spacing": 2}})
    spacing, views = config["video"]["view_spacing"], config["video"]["window_views"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for warp in WARPS:
            for clip_index in range(N_CLIPS):
                path = str(Path(tmp) / f"clip_{warp}_{clip_index}.mp4")
                render_synthetic_video(path, np.random.default_rng(SEED + clip_index), warp_amplitude=warp)
                clip = decode(path, 10, 150)
                for start in range(0, len(clip.frames) - (views - 1) * spacing, 4):
                    record = window_statistic([clip.frames[start + k * spacing] for k in range(views)],
                                              config, SEED, None)
                    rows.append({"Warp_Amplitude": warp, "Clip": clip_index, "Window_Start": start,
                                 "Used": record["used"], "Reason": record.get("reason", ""),
                                 "Statistic": record.get("statistic", np.nan),
                                 "N_Pencils": record.get("n_pencils", 0),
                                 "Median_Displacement_Frac": record.get("median_displacement_frac", np.nan)})
    df = pd.DataFrame(rows)
    print(df[df.Used].groupby("Warp_Amplitude")["Statistic"].describe()[["count", "50%", "min", "max"]])
    save_csv(df, "video_fixture.csv", SEED, "S6b: video branch on rendered clips", config)


if __name__ == "__main__":
    main()
