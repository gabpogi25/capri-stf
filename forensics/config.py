"""
forensics/config.py

Load the parameter registry (configs/default.yaml) and compute its hash.

The hash is recorded in every output so a number can be traced to the exact
parameter set that produced it.
"""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Path | str | None = None, overrides: dict[str, Any] | None = None) -> dict:
    """Read a YAML config and apply nested overrides such as {"lines": {"edge_sigma_px": 1.0}}."""
    with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if overrides:
        config = merge(config, overrides)
    return config


def merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def config_hash(config: dict) -> str:
    """SHA-256 of the canonical JSON form (sorted keys), first 16 hex digits."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
