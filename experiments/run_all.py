"""
experiments/run_all.py

Run every simulation experiment of the 3D-reconstruction study in order, then
the Confidence Fusion metrics report. Equivalent to `make experiments`.

Experiments that need external data or optional packages have their own runners:
    run_real_world_baselines.py   needs the IMC 2025 images, hloc and pycolmap
    run_real_images.py      needs the IMC 2025 images and PyTorch
    ml_models/*/train*.py         need PyTorch (their committed outputs are used)

Usage: python experiments/run_all.py [--only name1 name2 ...]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    "run_invariance.py",
    "run_stratification.py",       # rank classifier
    "run_conditioning.py",
    "run_kappa_conditioning.py",  # kappa_J vs L(S) (Section 5 of docs/MATHEMATICS.md)
    "run_syzygy_residual.py",
    "run_drift_admission.py",
    "run_pencil_drift.py",        # pencil drift test (Section 4 of docs/MATHEMATICS.md)
    "run_fusion_metrics.py",       # reads the committed neural-fusion evaluation file
    "run_ablation.py",
    "run_trust_calibration.py",
    "run_synthetic_baselines.py",
    "run_runtime.py",             # runtime appendix
    "run_pencil_runtime.py",     # runtime of kappa_J, pencil test, selector
    "run_scaling_law.py",
    "run_drift_transition.py",
    "run_residual_bound.py",
    "run_boundary_shape.py",
    "run_greedy_efficiency.py",
    "run_signal_combination.py",
    "run_sharpness_vs_n.py",
    "run_tuple_count.py",
    "run_separation_stability.py",
    "run_noise_amplification.py",
    "run_degeneracy_matched.py",  # degeneracy pre-registration; ~6 min
    "run_logistic_fusion.py",           # logistic fusion pre-registration; ~5-8 min
    "run_trust_calibrated.py",   # calibrated-trust pre-registration; ~7 min
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="run only these script names")
    args = parser.parse_args()
    experiments_dir = Path(__file__).resolve().parent
    selected = args.only or EXPERIMENTS
    failed = []
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        start = time.time()
        result = subprocess.run([sys.executable, str(experiments_dir / name)], cwd=experiments_dir)
        status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
        print(f"=== {name}: {status} in {time.time() - start:.0f} s ===", flush=True)
        if result.returncode != 0:
            failed.append(name)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
