"""
experiments/forensics/media_statistics.py

Compute the uncalibrated statistics of every item in a manifest (cached as a
CSV next to the results), shared by Gates A, B and C:
    image  -> mean z^2 over constraints (NaN when the branch abstains)
    video  -> per-window pencil drift statistics and per-frame image statistics
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR
from forensics.config import config_hash
from forensics.pipeline_image import analyze_image, image_statistic, read_gray
from forensics.pipeline_video import window_statistic
from forensics.video.decode import decode, split_shots


def image_table(manifest: pd.DataFrame, config: dict, seed: int) -> pd.DataFrame:
    rows = []
    for k, item in manifest.reset_index(drop=True).iterrows():
        evidence = analyze_image(item["abs_path"], config, seed + k)
        z = np.array([c.z for c in evidence.constraints])
        rows.append({**item.drop(labels=["abs_path"]).to_dict(), "decision_uncalibrated": evidence.decision,
                     "abstain": evidence.decision == "insufficient", "n_constraints": evidence.n_constraints,
                     "statistic": image_statistic(z) if evidence.decision != "insufficient" else np.nan})
    return pd.DataFrame(rows)


def _video_row(job: tuple) -> dict:
    """Statistics of one clip; job = (item as dict, config, seed of this clip, frame stride)."""
    from forensics.pipeline_image import analyze_gray

    item, config, clip_seed, frame_stride = job
    cfg_ingest, cfg = config["ingest"], config["video"]
    span = (cfg["window_views"] - 1) * cfg["view_spacing"] + 1
    clip = decode(item["abs_path"], cfg_ingest["video_sample_fps"], cfg_ingest["max_frames_per_clip"])
    windows, frames, kappas = [], [], []
    rng = np.random.default_rng(clip_seed)
    for start, end in split_shots(clip, cfg_ingest["shot_threshold"], cfg_ingest["min_shot_len_frames"]):
        for w0 in range(start, end - span + 1, span):
            record = window_statistic([clip.frames[w0 + j * cfg["view_spacing"]] for j in range(cfg["window_views"])],
                                      config, int(rng.integers(2 ** 31)), None)
            if record["used"]:
                windows.append(record["statistic"])
                kappas += record.get("kappa_values", [])
        for i in range(start, end, frame_stride):
            gray = clip.frames[i].mean(axis=2).astype(np.uint8)
            evidence = analyze_gray(gray, config, rng)
            if evidence.decision != "insufficient":
                frames.append(image_statistic(np.array([c.z for c in evidence.constraints])))
    return {**{k: v for k, v in item.items() if k != "abs_path"}, "window_statistics": json.dumps(windows),
            "frame_statistics": json.dumps(frames), "kappa_values": json.dumps(kappas[:2000]),
            "abstain": not windows and not frames, "codec": json.dumps(clip.metadata)}


def video_table(manifest: pd.DataFrame, config: dict, seed: int, frame_stride: int = 5,
                workers: int | None = None) -> pd.DataFrame:
    """One row per clip, in manifest order. Clip k uses seed + k whatever the number of workers,
    so the table does not depend on `workers` (tests/test_forensics_parallel.py)."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(item.to_dict(), config, seed + k, frame_stride)
            for k, item in manifest.reset_index(drop=True).iterrows()]
    workers = max(1, (os.cpu_count() or 2) - 1) if workers is None else workers
    if workers == 1 or len(jobs) < 2:
        return pd.DataFrame([_video_row(job) for job in jobs])
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"   # one BLAS thread per worker process
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame(list(pool.map(_video_row, jobs)))


def cached_table(manifest_path: Path, manifest: pd.DataFrame, config: dict, media: str) -> pd.DataFrame:
    """
    Statistics for EVERY item of the manifest, cached per (manifest checksum, config hash).
    The seed is config["global"]["base_seed"], so the real-media null and held-out detection gates and degradation sweep see identical
    statistics; callers filter rows afterwards.
    """
    from forensics.io.provenance import sha256_file

    key = f"{manifest_path.stem}_{sha256_file(manifest_path)[:12]}_{config_hash(config)}"
    cache = RESULTS_DIR / f"cache_{media}_{key}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"source": str, "split": str, "label": str})
    seed = int(config["global"]["base_seed"])
    table = image_table(manifest, config, seed) if media == "image" else video_table(manifest, config, seed)
    table.to_csv(cache, index=False)
    return table
