"""
forensics/io/manifests.py

Dataset manifests.

A manifest is a CSV with one row per media item:

    path,label,source,generator,split,sha256
    data/real/scene01/frame_0001.png,real,scene01,none,train_null,<sha256>
    data/fake/sd15/0001.png,generated,prompt_0001,sd15,test,<sha256>

    label      "real" or "generated"
    source     identifier of the underlying scene / prompt / source video;
               items sharing a source must be in the same split
    generator  generator name ("none" for real media)
    split      one of train_null, val, cal, test
    sha256     checksum of the file (verified on load unless verify=False)

Rules enforced on load: required columns present, labels and splits valid,
no source appears in two splits, files exist, checksums match.
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

from forensics.config import PROJECT_ROOT
from forensics.io.provenance import sha256_file

REQUIRED_COLUMNS = ["path", "label", "source", "generator", "split", "sha256"]
VALID_LABELS = {"real", "generated"}
VALID_SPLITS = {"train_null", "val", "cal", "test"}


class ManifestError(ValueError):
    """Raised when a manifest violates the rules above."""


def load_manifest(path: Path | str, verify: bool = True, root: Path | None = None) -> pd.DataFrame:
    """Load and validate a manifest; paths are resolved relative to `root` (default: project root)."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")
    df = pd.read_csv(path, comment="#", dtype=str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"{path}: missing columns {missing}")
    if df.empty:
        raise ManifestError(f"{path}: manifest has no rows")
    bad_labels = set(df["label"]) - VALID_LABELS
    if bad_labels:
        raise ManifestError(f"{path}: invalid labels {bad_labels}")
    bad_splits = set(df["split"]) - VALID_SPLITS
    if bad_splits:
        raise ManifestError(f"{path}: invalid splits {bad_splits}")
    splits_per_source = df.groupby("source")["split"].nunique()
    leaking = splits_per_source[splits_per_source > 1]
    if len(leaking):
        raise ManifestError(f"{path}: sources in more than one split: {list(leaking.index[:10])}")

    base = root or PROJECT_ROOT
    df["abs_path"] = [str((base / p).resolve()) if not Path(p).is_absolute() else p for p in df["path"]]
    missing_files = [p for p in df["abs_path"] if not Path(p).exists()]
    if missing_files:
        raise ManifestError(f"{path}: {len(missing_files)} files missing, e.g. {missing_files[:3]}")
    if verify:
        mismatched = [p for p, h in zip(df["abs_path"], df["sha256"]) if sha256_file(p) != h]
        if mismatched:
            raise ManifestError(f"{path}: {len(mismatched)} checksum mismatches, e.g. {mismatched[:3]}")
    df["y"] = (df["label"] == "generated").astype(int)
    return df


def write_manifest(rows: list[dict], path: Path | str, root: Path | None = None) -> pd.DataFrame:
    """Write a manifest, computing sha256 for every row from its file."""
    base = root or PROJECT_ROOT
    records = []
    for row in rows:
        file_path = Path(row["path"])
        absolute = file_path if file_path.is_absolute() else base / file_path
        records.append({**{k: row[k] for k in REQUIRED_COLUMNS if k != "sha256"},
                        "sha256": sha256_file(absolute)})
    df = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    df.to_csv(path, index=False)
    return df
