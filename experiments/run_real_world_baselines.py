"""
experiments/run_real_world_baselines.py

Real-image benchmark. Every reported number comes
from a logged run of this script, which is the only supported way to produce
the table.

Pipelines (all use COLMAP incremental mapping WITH geometric verification):
    sift            SIFT + NN-ratio matching (hloc), no extra filtering
    superglue       SuperPoint + SuperGlue, no extra filtering
    superglue_ransac        + per-pair RANSAC prefilter (control)
    superglue_ransac_capri  + RANSAC prefilter + CAPRI reliability layer

Protocol: several IMC 2025 scene units, several seeds per pipeline, one CSV row
per (dataset, scene, pipeline, seed). An IMC "dataset" can hold several scenes,
each with its own ground-truth frame, plus outlier images; the unit is therefore
one scene ('dataset/scene'), reconstructed from that scene's images only, with
outliers excluded. The largest COLMAP model is scored; a run with no model is
kept as a row (Failure column, zero registered). SuperGlue features and matches
are computed once per scene and shared by the three SuperGlue pipelines and all
seeds, so the pipelines differ only in filtering. Image data are read from the
user's local copy and are NOT redistributed (see DATA.md).

Metrics per run:
    n_images, n_registered                  images in the scene / registered by COLMAP
    median_pose_error_deg                   max(rotation, translation-direction) error
                                            over all pairs of registered images
    acc_at_{5,10,20}                        fraction of pairs with error below the threshold
    maa_at_{10}                             mean accuracy over thresholds 1..10 deg (area
                                            under the accuracy curve; IMC-style)
    ate_rmse_sim3_gt_units                  camera-centre RMSE after Sim(3) Umeyama
                                            alignment, in the units of the GT poses (IMC
                                            ground truth has arbitrary scale, so this is
                                            NOT metres)
    matches_in / matches_out / pairs_capri_empty   from the filtering stage
    wall-clock seconds per stage

Usage:
    python experiments/run_real_world_baselines.py \
        --dataset_dir data/real_images/image-matching-challenge-2025 \
        --scenes imc2023_haiper/fountain amy_gardens/peach \
        --seeds <seeds from the pre-registration>
    python experiments/run_real_world_baselines.py --dataset_dir <dir> --list_scenes
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import log_hardware_info, save_results  # noqa: E402

PIPELINES = ("sift", "superglue", "superglue_ransac", "superglue_ransac_capri")


OUTLIER_SCENE = "outliers"


def parse_unit(unit: str) -> tuple[str, str]:
    """'dataset/scene' -> (dataset, scene). Each scene has its own ground-truth frame."""
    assert unit.count("/") == 1, f"scene unit must be 'dataset/scene', got {unit!r}"
    dataset, scene = unit.split("/")
    assert scene != OUTLIER_SCENE, "outlier images have no common frame and are not a scene"
    return dataset, scene


def list_units(labels_csv: Path) -> list[str]:
    """Every 'dataset/scene' unit in the labels file, outliers excluded."""
    df = pd.read_csv(labels_csv)
    df = df[df["scene"] != OUTLIER_SCENE]
    return sorted(f"{d}/{s}" for d, s in df[["dataset", "scene"]].drop_duplicates().itertuples(index=False))


def stage_scene_images(dataset_dir: Path, labels_csv: Path, dataset: str, scene: str, dest: Path) -> int:
    """Copy one scene's images (no outliers, no other scenes) into `dest`; return the count."""
    df = pd.read_csv(labels_csv)
    names = sorted(df[(df["dataset"] == dataset) & (df["scene"] == scene)]["image"])
    assert names, f"no images labelled for {dataset}/{scene}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = dataset_dir / "train" / dataset / name
        if not (dest / name).exists():
            shutil.copy2(source, dest / name)
    return len(names)


def load_ground_truth(labels_csv: Path, dataset: str, scene: str) -> dict[str, dict[str, np.ndarray]]:
    """IMC train_labels.csv -> {image_name: {"R": 3x3, "t": (3,)}} for one scene."""
    df = pd.read_csv(labels_csv)
    df = df[(df["dataset"] == dataset) & (df["scene"] == scene)]
    poses = {}
    for _, row in df.iterrows():
        rotation = np.array([float(x) for x in str(row["rotation_matrix"]).split(";")]).reshape(3, 3)
        translation = np.array([float(x) for x in str(row["translation_vector"]).split(";")])
        poses[row["image"]] = {"R": rotation, "t": translation}
    return poses


def relative_pose_errors(estimated: dict, ground_truth: dict) -> np.ndarray:
    """max(rotation error, translation-direction error) in degrees for every ground-truth pair.

    A pair with an unregistered image gets an infinite error, so a pipeline is not
    rewarded for leaving hard images out (intention to treat, rule R4).
    """
    names = sorted(ground_truth)
    errors = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            if name_a not in estimated or name_b not in estimated:
                errors.append(np.inf)
                continue
            r_est = estimated[name_b]["R"] @ estimated[name_a]["R"].T
            t_est = estimated[name_b]["t"] - r_est @ estimated[name_a]["t"]
            r_gt = ground_truth[name_b]["R"] @ ground_truth[name_a]["R"].T
            t_gt = ground_truth[name_b]["t"] - r_gt @ ground_truth[name_a]["t"]
            cos_rot = np.clip((np.trace(r_est @ r_gt.T) - 1.0) / 2.0, -1.0, 1.0)
            rot_err = np.degrees(np.arccos(cos_rot))
            n_est, n_gt = np.linalg.norm(t_est), np.linalg.norm(t_gt)
            if n_est < 1e-9 or n_gt < 1e-9:
                trans_err = 180.0
            else:
                trans_err = np.degrees(np.arccos(np.clip(t_est @ t_gt / (n_est * n_gt), -1.0, 1.0)))
            errors.append(max(rot_err, trans_err))
    return np.asarray(errors, dtype=float)


def ate_rmse_sim3(estimated: dict, ground_truth: dict) -> float:
    """Camera-centre RMSE after Sim(3) Umeyama alignment (units of the GT poses)."""
    names = sorted(set(estimated) & set(ground_truth))
    if len(names) < 3:
        return float("nan")
    centre = lambda pose: -pose["R"].T @ pose["t"]  # noqa: E731
    est = np.stack([centre(estimated[n]) for n in names], axis=1)
    gt = np.stack([centre(ground_truth[n]) for n in names], axis=1)
    mu_est, mu_gt = est.mean(axis=1, keepdims=True), gt.mean(axis=1, keepdims=True)
    est_c, gt_c = est - mu_est, gt - mu_gt
    covariance = gt_c @ est_c.T / est.shape[1]
    u, d, vt = np.linalg.svd(covariance)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1.0
    rotation = u @ s @ vt
    var_est = np.mean(np.sum(est_c ** 2, axis=0))
    scale = np.trace(np.diag(d) @ s) / var_est
    aligned = scale * rotation @ est + (mu_gt - scale * rotation @ mu_est)
    return float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=0))))


def load_estimated_poses(sparse_model_dir: Path) -> dict:
    import pycolmap

    if not sparse_model_dir.exists():
        return {}
    reconstruction = pycolmap.Reconstruction(str(sparse_model_dir))
    poses = {}
    for image in reconstruction.images.values():
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        poses[image.name] = {"R": np.asarray(pose.rotation.matrix()), "t": np.asarray(pose.translation)}
    return poses


def run_sift(image_dir: Path, out: Path, match_dir: Path) -> dict:
    """SIFT + NN-ratio. Features and matches do not use the seed: computed once per scene."""
    import pycolmap
    from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction
    from colmap_capri_automator import keep_largest_model, single_process_loading
    from reconstruction.gpu_matching import match_scene

    t0 = time.perf_counter()
    match_dir.mkdir(parents=True, exist_ok=True)
    features, matches, pairs = match_dir / "features.h5", match_dir / "matches.h5", match_dir / "pairs.txt"
    done_marker = match_dir / "MATCHING_COMPLETE"
    if not done_marker.exists():
        for path in (features, matches, pairs):
            path.unlink(missing_ok=True)
        with single_process_loading():
            extract_features.main(extract_features.confs["sift"], image_dir, feature_path=features)
            pairs_from_exhaustive.main(pairs, features=features)
        match_scene(match_features.confs["NN-ratio"], pairs, features, matches)
        done_marker.touch()
    t_match = time.perf_counter() - t0
    t0 = time.perf_counter()
    reconstruction.main(out / "sparse", image_dir, pairs, features, matches,
                        camera_mode=pycolmap.CameraMode.AUTO, verbose=False)
    keep_largest_model(out / "sparse")
    return {"timings": {"matching_s": t_match, "sparse_s": time.perf_counter() - t0}}


def load_progress(path: Path) -> dict[tuple, dict]:
    """Finished rows of an interrupted run, keyed by (dataset, scene, pipeline, seed).

    Each line records the code commit; resuming under different code is refused, so a
    result file never mixes rows from two versions of the pipeline.
    """
    from forensics.io.provenance import git_commit

    if not path.exists():
        return {}
    commit, done = git_commit(), {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert row.pop("_commit") == commit, f"{path} was written by other code; delete it to start afresh"
        done[(row["Dataset"], row["Scene"], row["Pipeline"], row["Seed"])] = row
    return done


def append_progress(path: Path, row: dict) -> None:
    """Append one finished row (with the code commit) so an interrupted run can resume."""
    from forensics.io.provenance import git_commit

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({**row, "_commit": git_commit()}) + "\n")


def run_one(pipeline: str, image_dir: Path, out: Path, seed: int, match_dir: Path) -> dict:
    """One reconstruction. A failure (no model) is returned as a row, never dropped (R4)."""
    from colmap_capri_automator import run_superglue_pipeline

    out.mkdir(parents=True, exist_ok=True)
    try:
        if pipeline == "sift":
            return run_sift(image_dir, out, match_dir / "sift")
        mode = {"superglue": "none", "superglue_ransac": "ransac", "superglue_ransac_capri": "ransac_capri"}[pipeline]
        return run_superglue_pipeline(str(image_dir), str(out), mode=mode, seed=seed, match_dir=str(match_dir))
    except AssertionError as error:  # hloc asserts when COLMAP produces no model
        return {"failure": f"no model: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+",
                        help="'dataset/scene' units (see --list_scenes); outliers are never used")
    parser.add_argument("--batch", type=int,
                        help="run the scenes of this batch of results/real_reconstruction_batches.csv")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
    parser.add_argument("--output_root", type=Path, default=Path("outputs/benchmark_real"))
    parser.add_argument("--results_name", default="real_world_baselines.csv")
    if "--list_scenes" in sys.argv:
        dataset_dir = Path(sys.argv[sys.argv.index("--dataset_dir") + 1])
        print("\n".join(list_units(dataset_dir / "train_labels.csv")))
        return
    args = parser.parse_args()
    assert (args.scenes is None) != (args.batch is None), "give exactly one of --scenes and --batch"
    if args.batch is not None:
        batches = pd.read_csv(Path(__file__).parent / "results" / "real_reconstruction_batches.csv", comment="#")
        chosen = batches[batches["Batch"] == args.batch]
        assert len(chosen), f"batch {args.batch} is empty"
        args.scenes = [f"{d}/{s}" for d, s in zip(chosen["Dataset"], chosen["Scene"])]

    labels_csv = args.dataset_dir / "train_labels.csv"
    hardware = log_hardware_info()
    machine = {"Platform": hardware["platform"], "GPU": hardware.get("gpu_name", "none"),
               "Torch": hardware["torch_version"]}
    progress_path = args.output_root / "progress.jsonl"
    done = load_progress(progress_path)
    rows = []
    for unit in args.scenes:
        dataset, scene = parse_unit(unit)
        image_dir = args.output_root / "_images" / dataset / scene
        n_images = stage_scene_images(args.dataset_dir, labels_csv, dataset, scene, image_dir)
        ground_truth = load_ground_truth(labels_csv, dataset, scene)
        match_dir = args.output_root / "_matches" / dataset / scene
        for pipeline in args.pipelines:
            for seed in args.seeds:
                if (dataset, scene, pipeline, seed) in done:
                    rows.append(done[(dataset, scene, pipeline, seed)])
                    continue
                out = args.output_root / dataset / scene / pipeline / f"seed_{seed}"
                if out.exists():
                    shutil.rmtree(out)
                stats = run_one(pipeline, image_dir, out, seed, match_dir)
                estimated = load_estimated_poses(out / "sparse" / "0")
                errors = relative_pose_errors(estimated, ground_truth)
                thresholds = np.arange(1, 11)
                row = {
                    "Dataset": dataset, "Scene": scene, "Pipeline": pipeline, "Seed": seed,
                    "Failure": stats.get("failure", ""),
                    "N_Images": n_images, "N_Registered": len(estimated),
                    "N_Pose_Pairs": int(errors.size),
                    "Median_Pose_Error_deg": float(np.median(errors)) if errors.size else float("nan"),
                    "Acc_at_5": float(np.mean(errors < 5)) if errors.size else 0.0,
                    "Acc_at_10": float(np.mean(errors < 10)) if errors.size else 0.0,
                    "Acc_at_20": float(np.mean(errors < 20)) if errors.size else 0.0,
                    "mAA_at_10": float(np.mean([np.mean(errors < th) for th in thresholds])) if errors.size else 0.0,
                    "ATE_RMSE_Sim3_GT_Units": ate_rmse_sim3(estimated, ground_truth),
                    "Matches_In": stats.get("matches_in", np.nan),
                    "Matches_After_RANSAC": stats.get("matches_after_ransac", np.nan),
                    "Matches_Out": stats.get("matches_out", np.nan),
                    "Pairs_CAPRI_Empty": stats.get("pairs_capri_empty", np.nan),
                    "Pairs_RANSAC_Error": stats.get("pairs_ransac_error", np.nan),
                    **{f"Time_{k}": v for k, v in stats.get("timings", {}).items()},
                    **machine,
                }
                rows.append(row)
                append_progress(progress_path, row)
                print(json.dumps(row))
    save_results(pd.DataFrame(rows), args.results_name, min(args.seeds),
                 f"Real-image baselines on IMC 2025 scene units {args.scenes}")


if __name__ == "__main__":
    main()
