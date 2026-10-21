"""
forensics/io/provenance.py

Provenance record written next to every forensic output: git commit, config
hash, seed, library versions and the SHA-256 of every input file. A number in
a table must be traceable to one of these records.
"""

from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from forensics.config import PROJECT_ROOT, config_hash


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Output directories: a run rewrites these itself, so they must not mark the code as modified.
OUTPUT_PATHSPECS = [":(exclude)experiments/results", ":(exclude)figures"]


def git_commit() -> str:
    """Short HEAD commit, flagged when tracked SOURCE files differ from it."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".",
                                *OUTPUT_PATHSPECS], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + (" (with uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def library_versions() -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("scipy", "cv2", "sklearn", "pandas"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def provenance_record(config: dict, seed: int, inputs: list[Path | str] | None = None) -> dict:
    return {
        "git_commit": git_commit(),
        "config_hash": config_hash(config),
        "seed": int(seed),
        "versions": library_versions(),
        "inputs": {str(p): sha256_file(p) for p in (inputs or [])},
    }


def write_provenance(path: Path | str, config: dict, seed: int, inputs: list[Path | str] | None = None) -> dict:
    record = provenance_record(config, seed, inputs)
    Path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
