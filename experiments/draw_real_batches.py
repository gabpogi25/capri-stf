"""
experiments/draw_real_batches.py

Scene selection and batch draw for docs/prereg/real_reconstruction.md, run once before
any confirmatory reconstruction.

Rule: every IMC 2025 training scene (outlier images excluded) with at most
MAX_IMAGES images is included, with all its images. Batch 1 is BATCH1_SIZE included
scenes drawn at random with DRAW_SEED; batch 2 is the remaining included scenes;
the excluded scenes form the optional batch 3.

The estimated hours per scene are a planning figure only, extrapolated from the
smoke run (smoke_real_world_baselines_gpu.csv): per-pair matching and filtering
rates from imc2023_haiper/fountain, and a power law for COLMAP mapping time fitted
to the two smoke scenes. They are not runtime results.

Output: experiments/results/real_reconstruction_batches.csv (save_results).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402

MAX_IMAGES = 75
BATCH1_SIZE = 10
DRAW_SEED = 86_600_000
N_SEEDS = 3
N_PIPELINES = 4
OUTLIER_SCENE = "outliers"


def scene_sizes(labels_csv: Path) -> pd.DataFrame:
    """One row per scene unit: dataset, scene, number of labelled images (outliers excluded)."""
    df = pd.read_csv(labels_csv)
    df = df[df["scene"] != OUTLIER_SCENE]
    sizes = df.groupby(["dataset", "scene"]).size().rename("N_Images").reset_index()
    sizes.columns = ["Dataset", "Scene", "N_Images"]
    return sizes.sort_values(["Dataset", "Scene"]).reset_index(drop=True)


def draw_batches(sizes: pd.DataFrame, seed: int = DRAW_SEED) -> pd.DataFrame:
    """Batch per unit: 1 or 2 for included scenes (random draw), 3 for excluded ones."""
    out = sizes.copy()
    included = out.index[out["N_Images"] <= MAX_IMAGES].to_numpy()
    assert included.size > BATCH1_SIZE
    rng = np.random.default_rng(seed)
    batch1 = set(rng.choice(included, size=BATCH1_SIZE, replace=False).tolist())
    out["Included"] = out["N_Images"] <= MAX_IMAGES
    out["Batch"] = [1 if i in batch1 else (2 if inc else 3) for i, inc in zip(out.index, out["Included"])]
    return out


def estimate_hours(sizes: pd.DataFrame, smoke_csv: Path) -> pd.Series:
    """Planning estimate of hours per scene for 3 seeds x 4 pipelines (see module docstring)."""
    smoke = pd.read_csv(smoke_csv, comment="#")
    fountain = smoke[smoke["Scene"] == "fountain"].set_index("Pipeline")
    n_f = int(fountain["N_Images"].iloc[0])
    pairs_f = n_f * (n_f - 1) / 2
    per_pair = (fountain.loc["superglue", "Time_matching_s"] + fountain.loc["sift", "Time_matching_s"]
                + N_SEEDS * (fountain.loc["superglue_ransac", "Time_filtering_s"]
                             + fountain.loc["superglue_ransac_capri", "Time_filtering_s"])) / pairs_f
    by_scene = smoke.groupby("Scene").agg(n=("N_Images", "first"), t=("Time_sparse_s", "mean"))
    assert len(by_scene) == 2, "the mapping fit uses exactly the two smoke scenes"
    (n_a, t_a), (n_b, t_b) = by_scene.sort_values("n").to_numpy()
    exponent = np.log(t_b / t_a) / np.log(n_b / n_a)
    n = sizes["N_Images"].to_numpy(dtype=float)
    mapping = N_SEEDS * N_PIPELINES * t_b * (n / n_b) ** exponent
    return pd.Series((n * (n - 1) / 2 * per_pair + mapping) / 3600, index=sizes.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--smoke_csv", type=Path, default=RESULTS_DIR / "smoke_real_world_baselines_gpu.csv")
    args = parser.parse_args()

    sizes = scene_sizes(args.dataset_dir / "train_labels.csv")
    table = draw_batches(sizes)
    table["Est_Hours_Planning"] = estimate_hours(sizes, args.smoke_csv).round(2)
    table = table.sort_values(["Batch", "Dataset", "Scene"]).reset_index(drop=True)
    save_results(table, "real_reconstruction_batches.csv", DRAW_SEED,
                 f"Scene selection (<= {MAX_IMAGES} images) and batch draw for real_reconstruction")
    print(table.to_string())
    print(table.groupby("Batch").agg(Scenes=("Scene", "size"), Images=("N_Images", "sum"),
                                     Est_Hours=("Est_Hours_Planning", "sum")).round(1).to_string())


if __name__ == "__main__":
    main()
