"""
experiments/run_real_runtime.py

Runtime of the real-image pipelines on the evaluation machine, per stage, on GPU and on CPU.

Runs experiments/run_real_world_baselines.py twice on the same small scene units, one seed,
all four pipelines, each time in a fresh output folder (so features and matches are
computed, not reused):
    gpu   PyTorch sees the GPU (SuperPoint, SuperGlue and NN-ratio matching on it)
    cpu   CUDA_VISIBLE_DEVICES=-1 hides the GPU, so every stage runs on the CPU
The RANSAC/CAPRI filtering and COLMAP mapping run on the CPU in both. Each run writes its
rows (with the Time_* columns, platform, GPU and PyTorch version) through save_results to
real_runtime_{gpu,cpu}.csv.

The machine must be otherwise idle: no download or other heavy job.
These are timings, not accuracy results; the seed is RUNTIME_SEED.
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITS = ["ETs/ET", "imc2023_haiper/bike"]
RUNTIME_SEED = 86_950_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path,
                        default=ROOT / "data/kagglehub-cache/competitions/image-matching-challenge-2025")
    parser.add_argument("--devices", nargs="+", choices=("gpu", "cpu"), default=["gpu", "cpu"])
    args = parser.parse_args()
    for device in args.devices:
        out = ROOT / "outputs" / f"runtime_{device}"
        if out.exists():
            shutil.rmtree(out)
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        command = [sys.executable, str(ROOT / "experiments/run_real_world_baselines.py"),
                   "--dataset_dir", str(args.dataset_dir), "--scenes", *UNITS,
                   "--seeds", str(RUNTIME_SEED), "--output_root", str(out),
                   "--results_name", f"real_runtime_{device}.csv"]
        print(f"== {device}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env, cwd=ROOT)


if __name__ == "__main__":
    main()
