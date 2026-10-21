"""
synthetic_data/regimes.py

Top-level dataset generation script for all five CAPRI validation regimes.

Run directly:
    python synthetic_data/generator_script.py --seed 42 --n_configs 5000

This script generates one .npz file per regime in data/.
It must be run before any training or evaluation.
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
    generate_sparse_view_configuration,
    generate_mismatched_configuration,
    generate_non_rigid_configuration,
)
from synthetic_data.export import save_dataset


def generate_all_regimes(
    n_configs: int,
    n_points: int,
    seed: int,
    scale: float = 1.0
) -> None:
    """
    Generate synthetic datasets for all five validation regimes.

    Args:
        n_configs: Number of configurations per regime.
        n_points: Number of points per configuration.
        seed: Base random seed (each regime uses seed + offset for independence).
        scale: Bounding box scale for point generation.
    """
    print(f"Generating {n_configs} configurations per regime, {n_points} points each.")
    print(f"Base seed: {seed}")

    regimes = [
        ("generic",        0),
        ("near_degenerate", 1),
        ("sparse_view",    2),
        ("mismatched",     3),
        ("non_rigid",      4),
    ]

    for regime_name, seed_offset in regimes:
        rng = np.random.default_rng(seed + seed_offset * 100000)
        points_list = []
        labels = []

        print(f"  Generating regime: {regime_name}")
        for _ in tqdm(range(n_configs), desc=regime_name):

            if regime_name == "generic":
                pts = generate_generic_configuration(n_points, scale, rng)
                points_list.append(pts)
                labels.append("generic")

            elif regime_name == "near_degenerate":
                frac = rng.uniform(0.4, 0.8)
                pts = generate_near_degenerate_configuration(
                    n_points, coplanar_fraction=frac,
                    plane_offset=rng.uniform(1e-4, 1e-2),
                    scale=scale, rng=rng
                )
                points_list.append(pts)
                labels.append("near_degenerate")

            elif regime_name == "sparse_view":
                n_views = int(rng.integers(2, 5))
                pts, visibility = generate_sparse_view_configuration(
                    n_points, n_views=n_views,
                    points_per_view=max(6, n_points // n_views),
                    scale=scale, rng=rng
                )
                points_list.append(pts)
                labels.append("sparse_view")

            elif regime_name == "mismatched":
                frac = rng.uniform(0.1, 0.3)
                pts, mismatch_idx = generate_mismatched_configuration(
                    n_points, mismatch_fraction=frac, scale=scale, rng=rng
                )
                points_list.append(pts)
                labels.append("mismatched")

            elif regime_name == "non_rigid":
                bound = rng.uniform(0.01, 0.1)
                pts_orig, pts_def, def_bound = generate_non_rigid_configuration(
                    n_points, deformation_bound=bound, scale=scale, rng=rng
                )
                # Store deformed version; deformation_bound tracked in metadata
                points_list.append(pts_def)
                labels.append("non_rigid")

        metadata = {
            "n_points": n_points,
            "scale": scale,
            "n_configs": n_configs,
        }
        path = save_dataset(
            dataset_name=f"{regime_name}_{n_configs}",
            points_list=points_list,
            labels=labels,
            metadata=metadata,
            seed=seed + seed_offset * 100000
        )
        print(f"    Saved: {path}")

    print("All regimes generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CAPRI synthetic datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_configs", type=int, default=5000)
    parser.add_argument("--n_points", type=int, default=12)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    generate_all_regimes(
        n_configs=args.n_configs,
        n_points=args.n_points,
        seed=args.seed,
        scale=args.scale
    )