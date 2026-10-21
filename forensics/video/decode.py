"""
forensics/video/decode.py

Video ingest: decode at NATIVE resolution (never
resample before geometric analysis), sample frames at video_sample_fps up to
max_frames_per_clip, split shots, and record codec metadata as covariates.
"""

from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DecodedClip:
    frames: list[np.ndarray]            # BGR uint8, native resolution
    frame_indices: list[int]            # indices in the original stream
    fps: float
    width: int
    height: int
    metadata: dict = field(default_factory=dict)


def probe(path: Path | str) -> dict:
    """ffprobe metadata (codec, bitrate, fps, size); empty dict if ffprobe is unavailable."""
    try:
        output = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,width,height,avg_frame_rate,bit_rate,nb_frames,pix_fmt:format=duration,size",
             "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
        return json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def decode(path: Path | str, sample_fps: float, max_frames: int) -> DecodedClip:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    frames, indices, index = [], [], 0
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(frame)
            indices.append(index)
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    height, width = frames[0].shape[:2]
    return DecodedClip(frames, indices, float(fps), width, height, probe(path))


def shot_boundaries(frames: list[np.ndarray], threshold: float) -> list[int]:
    """Indices (into frames) where a new shot starts, by HSV-histogram difference (x100)."""
    import cv2

    starts, previous = [0], None
    for i, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]).ravel()
        histogram /= max(histogram.sum(), 1.0)
        if previous is not None and 100.0 * 0.5 * np.abs(histogram - previous).sum() > threshold:
            starts.append(i)
        previous = histogram
    return starts


def split_shots(clip: DecodedClip, threshold: float, min_len_frames: int) -> list[tuple[int, int]]:
    """(start, end) index ranges of shots with at least min_len_frames ORIGINAL frames."""
    starts = shot_boundaries(clip.frames, threshold) + [len(clip.frames)]
    shots = []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a >= 2 and clip.frame_indices[b - 1] - clip.frame_indices[a] + 1 >= min_len_frames:
            shots.append((a, b))
    return shots
