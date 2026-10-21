"""
colmap_capri_automator.py

Real-image pipeline: SuperPoint + SuperGlue matching (via hloc), optional
match filtering, COLMAP sparse reconstruction, and optional dense MVS.

Match-filtering modes (the variable under test in the real-image benchmark)
-----------------------------------------------------------
    "none"          SuperGlue matches passed to COLMAP unchanged
    "ransac"        per-pair RANSAC fundamental-matrix inliers only (control)
    "ransac_capri"  RANSAC inliers, then the CAPRI reliability layer keeps a
                    match only if its triangulated point lies in an accepted
                    six-point tuple

Every mode runs with COLMAP geometric verification ON, and the "ransac"
control isolates what CAPRI adds on top of the RANSAC prefilter, so a
difference between modes is not mixed with a change in verification.

What CAPRI removes: individual 2D matches. Per image
pair, the RANSAC inliers are triangulated with projective cameras
P1 = [I | 0], P2 from F (no calibration), CAPRI selects and certifies
six-point tuples on those 3D points, and only matches whose points appear in
an accepted tuple are written back. If CAPRI accepts no tuple for a pair, the
policy `empty_pair_policy` decides: "keep_ransac" (default) keeps the RANSAC
inliers for that pair, "drop" removes all its matches. Both counts are logged
in filter_stats.json so the effect of the policy is visible.

COLMAP binary for the dense stage: set the COLMAP_EXE environment variable,
otherwise colmap/COLMAP-3.9.1-windows-cuda/bin/colmap.exe (bundled Windows
build) or `colmap` on PATH is used.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from reconstruction.match_filtering import FILTER_MODES, filter_matches  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_colmap_executable() -> str:
    """COLMAP binary: $COLMAP_EXE, then the bundled Windows build, then PATH."""
    if os.environ.get("COLMAP_EXE"):
        return os.environ["COLMAP_EXE"]
    bundled = PROJECT_ROOT / "colmap" / "COLMAP-3.9.1-windows-cuda" / "bin" / "colmap.exe"
    if bundled.exists():
        return str(bundled)
    found = shutil.which("colmap")
    if found is None:
        raise FileNotFoundError("COLMAP not found. Set COLMAP_EXE or install COLMAP.")
    return found


def run_command(cmd: list[str]) -> None:
    """Run a subprocess, streaming output; the bundled build needs its lib dir on PATH."""
    print(f"\n[RUNNING] {' '.join(cmd)}")
    env = os.environ.copy()
    bundled_lib = PROJECT_ROOT / "colmap" / "COLMAP-3.9.1-windows-cuda" / "lib"
    if bundled_lib.exists():
        env["PATH"] = str(bundled_lib) + os.pathsep + env.get("PATH", "")
    subprocess.run(cmd, check=True, env=env)


@contextmanager
def single_process_loading():
    """Force num_workers=0 in torch DataLoaders while hloc runs.

    hloc's matcher starts five loader worker processes; on Windows (spawn start
    method) they hang with no output. Loading in the main process gives the same
    matches; it is the loading, not the matching, that changes.
    """
    import torch.utils.data

    original = torch.utils.data.DataLoader

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return original(*args, **kwargs)

    torch.utils.data.DataLoader = loader
    try:
        yield
    finally:
        torch.utils.data.DataLoader = original


def extract_and_match_superglue(image_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """SuperPoint features + exhaustive SuperGlue matching via hloc."""
    from hloc import extract_features, match_features, pairs_from_exhaustive

    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs.txt"
    from reconstruction.gpu_matching import match_scene

    with single_process_loading():
        extract_features.main(extract_features.confs["superpoint_max"], image_dir, feature_path=features_path)
        pairs_from_exhaustive.main(pairs_path, features=features_path)
    # same model and output as hloc's match_features, with the features held on the GPU
    match_scene(match_features.confs["superglue"], pairs_path, features_path, matches_path)
    return features_path, matches_path, pairs_path


def reconstruct_sparse(image_dir: Path, pairs_path: Path, features_path: Path,
                       matches_path: Path, sparse_dir: Path):
    """COLMAP incremental mapping via hloc, WITH COLMAP geometric verification."""
    import pycolmap
    from hloc import reconstruction

    sparse_dir.mkdir(parents=True, exist_ok=True)
    model = reconstruction.main(
        sfm_dir=sparse_dir, image_dir=image_dir, pairs=pairs_path,
        features=features_path, matches=matches_path,
        camera_mode=pycolmap.CameraMode.AUTO, verbose=False,
    )
    keep_largest_model(sparse_dir)
    return model


def keep_largest_model(sfm_dir: Path) -> None:
    """Copy the largest model, which hloc moves into `sfm_dir` itself, to `sfm_dir/0`.

    hloc leaves `sfm_dir/models/<largest>` empty after the move, so `models/0` is
    either empty or a smaller model and must not be used.
    """
    model_files = [p for p in sfm_dir.glob("*.bin") if p.stem in ("cameras", "images", "points3D")]
    if not model_files:
        return
    target = sfm_dir / "0"
    target.mkdir(exist_ok=True)
    for path in model_files:
        shutil.copy2(path, target / path.name)


def reconstruct_dense(image_dir: Path, sparse_dir: Path, dense_dir: Path) -> Path:
    """Undistortion (pycolmap), PatchMatch stereo, fusion and Poisson meshing (COLMAP)."""
    import pycolmap

    dense_dir.mkdir(parents=True, exist_ok=True)
    pycolmap.undistort_images(output_path=str(dense_dir), input_path=str(sparse_dir / "0"),
                              image_path=str(image_dir))
    colmap_exe = resolve_colmap_executable()
    run_command([colmap_exe, "patch_match_stereo", "--workspace_path", str(dense_dir),
                 "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"])
    run_command([colmap_exe, "stereo_fusion", "--workspace_path", str(dense_dir),
                 "--workspace_format", "COLMAP", "--input_type", "geometric",
                 "--output_path", str(dense_dir / "fused.ply")])
    mesh_path = dense_dir / "meshed.ply"
    run_command([colmap_exe, "poisson_mesher", "--input_path", str(dense_dir / "fused.ply"),
                 "--output_path", str(mesh_path), "--PoissonMeshing.trim", "0"])
    return mesh_path


def run_superglue_pipeline(image_dir: str, output_dir: str, mode: str = "ransac_capri",
                           seed: int = 42, dense: bool = False, match_dir: str | None = None,
                           **filter_kwargs) -> dict:
    """SuperGlue matching -> filtering in `mode` -> COLMAP sparse (and optionally dense).

    With `match_dir`, features and matches are computed once there and reused by every
    later call with the same directory, so all modes and seeds see identical matches.
    """
    image_dir_path = Path(image_dir).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    t0 = time.perf_counter()
    match_out = Path(match_dir).resolve() if match_dir else out
    match_out.mkdir(parents=True, exist_ok=True)
    cached = [match_out / n for n in ("features.h5", "matches.h5", "pairs.txt")]
    done_marker = match_out / "MATCHING_COMPLETE"
    if match_dir and done_marker.exists() and all(p.exists() for p in cached):
        features_path, matches_path, pairs_path = cached
    else:
        for path in cached + [done_marker]:
            path.unlink(missing_ok=True)
        features_path, matches_path, pairs_path = extract_and_match_superglue(image_dir_path, match_out)
        done_marker.touch()
    timings["matching_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    filtered_path = out / f"matches_{mode}.h5"
    stats = filter_matches(features_path, matches_path, filtered_path, mode, seed=seed, **filter_kwargs)
    timings["filtering_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reconstruct_sparse(image_dir_path, pairs_path, features_path, filtered_path, out / "sparse")
    timings["sparse_s"] = time.perf_counter() - t0

    if dense and (out / "sparse" / "0").exists():
        t0 = time.perf_counter()
        reconstruct_dense(image_dir_path, out / "sparse", out / "dense")
        timings["dense_s"] = time.perf_counter() - t0

    stats["timings"] = timings
    with open(out / "filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


# Entry point used by dashboard.py; same as mode="ransac_capri".
def automate_superglue_capri_pipeline(image_dir: str, output_dir: str, log_callback=None) -> dict:
    return run_superglue_pipeline(image_dir, output_dir, mode="ransac_capri")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=FILTER_MODES, default="ransac_capri")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--empty_pair_policy", choices=("keep_ransac", "drop"), default="keep_ransac")
    args = parser.parse_args()
    print(json.dumps(run_superglue_pipeline(args.image_dir, args.output_dir, args.mode, args.seed,
                                            args.dense, empty_pair_policy=args.empty_pair_policy), indent=2))
