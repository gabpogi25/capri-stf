"""
forensics/eval/audit_metadata.py

Metadata-only shortcut audit (experiment E05, the held-out image gate).

A classifier that sees ONLY format metadata (resolution, file size, bitrate,
codec, frame rate, duration, JPEG quality estimate) is trained to separate
real from generated media with grouped cross-validation. If its AUC exceeds
the pre-registered bound (0.55), the dataset leaks format cues and any image
or video result on it is suspect until both classes are re-encoded through the
same pipeline.
"""

from __future__ import annotations
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def image_metadata(path: Path) -> dict:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    return {"width": width, "height": height, "aspect": width / height,
            "bytes_per_pixel": path.stat().st_size / (width * height),
            "is_jpeg": float(path.suffix.lower() in {".jpg", ".jpeg"}),
            "is_png": float(path.suffix.lower() == ".png"), "channels": image.shape[2] if image.ndim == 3 else 1}


def video_metadata(path: Path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,codec_name,avg_frame_rate,bit_rate,nb_frames:format=duration,size",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(output)
    stream, fmt = info["streams"][0], info["format"]
    num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
    duration = float(fmt.get("duration") or 0.0)
    return {"width": stream["width"], "height": stream["height"],
            "fps": float(num) / float(den) if float(den) else 0.0, "duration": duration,
            "bitrate": float(stream.get("bit_rate") or (8 * float(fmt["size"]) / max(duration, 1e-6))),
            "is_h264": float(stream.get("codec_name") == "h264"), "is_hevc": float(stream.get("codec_name") == "hevc")}


def metadata_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for path in manifest["abs_path"]:
        p = Path(path)
        rows.append(video_metadata(p) if p.suffix.lower() in VIDEO_SUFFIXES else image_metadata(p))
    return pd.DataFrame(rows).fillna(0.0)


def metadata_audit(manifest: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> dict:
    """Grouped-CV AUC of a logistic regression on metadata only (groups = source)."""
    features = metadata_table(manifest).to_numpy(dtype=float)
    labels = manifest["y"].to_numpy()
    groups = manifest["source"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    scores = np.zeros(len(labels))
    for train, test in GroupKFold(n_splits=n_splits).split(features, labels, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        model.fit(features[train], labels[train])
        scores[test] = model.predict_proba(features[test])[:, 1]
    return {"metadata_auc": float(roc_auc_score(labels, scores)), "n": int(len(labels)),
            "n_groups": int(len(np.unique(groups)))}
