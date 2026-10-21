"""The video statistics table does not depend on the number of worker processes."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "experiments" / "forensics")]

from forensics.config import load_config  # noqa: E402
from media_statistics import video_table  # noqa: E402


def _write_clip(path: Path, seed: int, n_frames: int = 40) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    base = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    for t in range(n_frames):
        writer.write(np.roll(base, shift=2 * t, axis=1))   # a sideways pan
    writer.release()


def test_video_table_is_the_same_serial_and_parallel(tmp_path: Path) -> None:
    paths = []
    for k in range(3):
        path = tmp_path / f"clip{k}.mp4"
        _write_clip(path, seed=k)
        paths.append(str(path))
    manifest = pd.DataFrame({"path": paths, "abs_path": paths, "label": "real", "source": [f"s{k}" for k in range(3)],
                             "generator": "none", "split": "val", "sha256": "x"})
    config = load_config()
    serial = video_table(manifest, config, seed=11, workers=1)
    parallel = video_table(manifest, config, seed=11, workers=2)
    assert serial.equals(parallel)
    assert list(serial["path"]) == paths   # manifest order is kept
