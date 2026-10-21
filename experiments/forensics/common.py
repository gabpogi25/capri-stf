"""
experiments/forensics/common.py

Shared output helpers for the forensic experiments: every CSV gets a header
with the seed, config hash and code commit, and a provenance JSON is written
beside it.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forensics.config import config_hash  # noqa: E402
from forensics.io.provenance import provenance_record  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "forensics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, name: str, seed: int, description: str, config: dict,
             inputs: list | None = None) -> Path:
    record = provenance_record(config, seed, inputs)
    path = RESULTS_DIR / name
    with open(path, "w", newline="") as f:
        f.write(f"# Experiment: {description}\n# Seed: {seed}\n# Config hash: {config_hash(config)}\n")
        f.write(f"# Code commit: {record['git_commit']}\n")
        df.to_csv(f, index=False)
    path.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2))
    print(f"  Saved: {path}")
    return path


def save_json(payload: dict, name: str, seed: int, config: dict, inputs: list | None = None) -> Path:
    path = RESULTS_DIR / name
    payload = {**payload, "provenance": provenance_record(config, seed, inputs)}
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"  Saved: {path}")
    return path


def require_manifest(path: str | None, purpose: str) -> Path:
    """Exit with a clear message (never substitute data) when a manifest is missing."""
    if not path or not Path(path).exists():
        sys.exit(f"{purpose} needs a dataset manifest (see data/manifests/README.md); got {path!r}. "
                 "No result is produced without real data.")
    return Path(path)
