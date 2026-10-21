"""
scripts/draw_forensic_batches.py

Batch draw for docs/prereg/forensics_video.md, run once before any confirmatory forensic run.

From a full manifest (scripts/build_manifest.py), batch 1 takes, at random with DRAW_SEED and
without replacement, the counts in BATCH1 per (split, label, generator) cell (a cell with
fewer items contributes all of them); batch 2 is every other item. Items are drawn by source,
so a source never spans two batches (one clip per source here). Writes the two subset
manifests next to the full one (data/manifests/, git-ignored, same columns) and a summary
of counts per batch and cell to experiments/results/forensics_video_batches.csv (save_results).

Usage:
  python scripts/draw_forensic_batches.py --manifest data/manifests/videos_v2.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from experiment_utils import save_results  # noqa: E402

DRAW_SEED = 88_600_000
# (split, label, generator or "*" for every generator of the cell) -> batch-1 count
BATCH1 = {
    ("train_null", "real", "none"): 300,
    ("val", "real", "none"): 300,
    ("test", "real", "none"): 300,
    ("cal", "generated", "*"): 300,
    ("test", "generated", "gen2"): 150,
    ("test", "generated", "moonvalley"): 150,
    ("test", "generated", "sora"): 150,
}


def draw(manifest: pd.DataFrame, seed: int = DRAW_SEED) -> pd.Series:
    """Batch (1 or 2) of every row of the manifest."""
    rng = np.random.default_rng(seed)
    batch = pd.Series(2, index=manifest.index)
    for (split, label, generator), count in BATCH1.items():
        cell = manifest[(manifest["split"] == split) & (manifest["label"] == label)]
        if generator != "*":
            cell = cell[cell["generator"] == generator]
        sources = np.array(sorted(cell["source"].unique()))
        assert len(sources), f"empty cell {split}/{label}/{generator}"
        chosen = set(rng.choice(sources, size=min(count, len(sources)), replace=False).tolist())
        batch[cell.index[cell["source"].isin(chosen)]] = 1
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, dtype=str)
    batch = draw(manifest)
    for b in (1, 2):
        out = args.manifest.with_name(f"{args.manifest.stem}_batch{b}.csv")
        manifest[batch == b].to_csv(out, index=False)
        print(f"batch {b}: {int((batch == b).sum())} items -> {out}")
    summary = (manifest.assign(Batch=batch).groupby(["Batch", "split", "label", "generator"]).size()
               .rename("N_Items").reset_index())
    save_results(summary, "forensics_video_batches.csv", DRAW_SEED,
                 f"Forensic video batch draw from {args.manifest.name}")
    print(summary[summary["Batch"] == 1].to_string(index=False))


if __name__ == "__main__":
    main()
