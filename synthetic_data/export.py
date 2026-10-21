"""
synthetic_data/export.py

Dataset serialization for CAPRI synthetic configurations.

Saves and loads generated point configurations and their metadata
to/from .npy and .npz files in the data/ directory.
Data generation and simulation execution are strictly separated:
generators are never called inside training or evaluation loops.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).parent.parent / "data"


def save_dataset(
    dataset_name: str,
    points_list: list[np.ndarray],
    labels: list[str],
    metadata: dict[str, Any],
    seed: int
) -> Path:
    """
    Save a list of point configurations and their regime labels to disk.

    Args:
        dataset_name: Identifier for this dataset (e.g., 'generic_5000').
        points_list: List of (4, N) homogeneous coordinate arrays.
        labels: List of regime label strings, one per configuration.
        metadata: Dictionary of dataset-level metadata (seed, thresholds, etc.).
        seed: Random seed used to generate this dataset.

    Returns:
        Path to the saved .npz file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{dataset_name}_seed{seed}.npz"

    # Stack into a single array for efficient storage
    # Pad to uniform size if needed
    max_n = max(arr.shape[1] for arr in points_list)
    n_configs = len(points_list)
    padded = np.zeros((n_configs, 4, max_n), dtype=np.float64)
    sizes = np.zeros(n_configs, dtype=np.int32)
    for k, arr in enumerate(points_list):
        n = arr.shape[1]
        padded[k, :, :n] = arr
        sizes[k] = n

    labels_array = np.array(labels, dtype='U32')
    metadata['seed'] = seed
    metadata['dataset_name'] = dataset_name

    np.savez(
        out_path,
        points=padded,
        sizes=sizes,
        labels=labels_array,
        **{f"meta_{k}": np.array(v) for k, v in metadata.items() if np.isscalar(v)}
    )
    return out_path


def load_dataset(dataset_path: Path) -> tuple[list[np.ndarray], list[str], dict]:
    """
    Load a saved dataset from a .npz file.

    Args:
        dataset_path: Path to the .npz file.

    Returns:
        Tuple of (points_list, labels, metadata_dict).
    """
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"
    data = np.load(dataset_path, allow_pickle=False)

    padded = data['points']       # (N_configs, 4, max_n)
    sizes = data['sizes']         # (N_configs,)
    labels = list(data['labels'])

    points_list = [
        padded[k, :, :sizes[k]]
        for k in range(len(sizes))
    ]

    metadata = {
        k.replace('meta_', ''): float(data[k])
        for k in data.files
        if k.startswith('meta_')
    }

    return points_list, labels, metadata