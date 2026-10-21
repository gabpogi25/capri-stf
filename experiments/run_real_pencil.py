"""
experiments/run_real_pencil.py

Pencil test between view subsets on real scenes (docs/prereg/real_pencil.md).

Tracks come from the IMC 2025 reconstructions of the benchmark (pipeline superglue_ransac,
seed 86,000,000; docs/prereg/real_reconstruction.md). A window is six registered views of
one scene and N_POINTS 3D points observed in all six, found by a randomized greedy search
over covisibility. Observations are undistorted with the COLMAP camera (pinhole afterwards)
and mapped to a 1280-pixel long side, the frame of the simulations
(synthetic_data/injection.py); every pixel figure here is in that frame.

Every window is scored unperturbed ("none") and once per injection family. An injected
variant adds to the REAL observations the displacement that the family induces on the
reprojected COLMAP points (synthetic_data.injection.make_window with the window's cameras
and points, no noise), so the real noise is kept. Views are ordered by image name; subsets
are A = {0, 2, 4}, B = {1, 3, 5} (experiments/eval/mv_window.py).

Statistics: experiments/eval/mv_window.window_statistics, unchanged, with noise_px = 1, so
split_glrt is the split sum of squares in px^2; the analysis divides it by sigma^2 estimated
on the calibration scenes.

Modes:
  --split    draw calibration and test scenes (seed SPLIT_SEED) -> real_pencil_split.csv
  --batch B  compute the windows of the scenes of reconstruction batch B
             -> real_pencil_windows_batchB.csv
"""

from __future__ import annotations
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "eval"))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402

N_POINTS = 30
N_VIEWS = 6
MAX_WINDOWS_PER_SCENE = 80
MAX_ATTEMPTS_PER_WINDOW = 25
TOP_CANDIDATES = 3
LONG_SIDE = 1280.0
FRAME_CENTRE = np.array([640.0, 360.0])
INJECTIONS = {"jitter": 2.0, "drift": 4.0, "warp": 2.0, "swap": 0.05}
SPLIT_SEED = 87_000_000
WINDOW_SEED = 87_100_000      # + 1000 * scene index + window index
SMOKE_OFFSET = 800_000        # exploration runs: 87.9e6 + ...
SOURCE_PIPELINE, SOURCE_SEED = "superglue_ransac", 86_000_000


def scene_family(dataset: str) -> str:
    """Scene family for the transfer question: the IMC source collection of the dataset."""
    for prefix in ("pt_", "imc2023_", "imc2024_"):
        if dataset.startswith(prefix):
            return prefix.rstrip("_")
    return dataset


def draw_split(batches: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Half of the included scenes (rounded down) are calibration scenes, drawn at random."""
    included = batches[batches["Included"]].sort_values(["Dataset", "Scene"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    cal = set(rng.choice(len(included), size=len(included) // 2, replace=False).tolist())
    included["Role"] = ["cal" if i in cal else "test" for i in range(len(included))]
    included["Family"] = included["Dataset"].map(scene_family)
    return included[["Dataset", "Scene", "N_Images", "Batch", "Role", "Family"]]


def load_scene(model_dir: Path) -> dict:
    """Pinhole cameras in the 1280-px frame, undistorted observations, and 3D points."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    views = {}
    for image_id, image in rec.images.items():
        if not image.has_pose:
            continue
        camera = rec.cameras[image.camera_id]
        scale = LONG_SIDE / max(camera.width, camera.height)
        to_frame = np.array([[scale, 0, FRAME_CENTRE[0] - scale * camera.width / 2],
                             [0, scale, FRAME_CENTRE[1] - scale * camera.height / 2], [0, 0, 1.0]])
        K = to_frame @ camera.calibration_matrix()
        pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        P = K @ np.asarray(pose.matrix())
        observed = [(p.point3D_id, p.xy) for p in image.points2D if p.has_point3D()]
        if not observed:
            continue
        ids = np.array([i for i, _ in observed])
        normalized = np.asarray(camera.cam_from_img(np.array([xy for _, xy in observed], dtype=float)))
        pixels = np.c_[normalized, np.ones(len(normalized))] @ K.T
        views[image_id] = {"name": image.name, "P": P,
                           "obs": dict(zip(ids.tolist(), pixels[:, :2] / pixels[:, 2:3]))}
    points = {pid: np.asarray(p.xyz) for pid, p in rec.points3D.items()}
    return {"views": views, "points": points}


def find_windows(scene: dict, rng: np.random.Generator, max_windows: int) -> list[tuple[tuple, np.ndarray]]:
    """Distinct six-view sets with >= N_POINTS common points, and N_POINTS of them drawn at random."""
    ids = sorted(scene["views"])
    visible = {i: set(scene["views"][i]["obs"]) for i in ids}
    windows, seen = [], set()
    for _ in range(max_windows * MAX_ATTEMPTS_PER_WINDOW):
        if len(windows) == max_windows or len(ids) < N_VIEWS:
            break
        chosen = [ids[rng.integers(len(ids))]]
        common = set(visible[chosen[0]])
        while len(chosen) < N_VIEWS:
            ranked = sorted(((len(common & visible[j]), j) for j in ids if j not in chosen), reverse=True)
            _, pick = ranked[rng.integers(min(TOP_CANDIDATES, len(ranked)))]
            chosen.append(pick)
            common &= visible[pick]
        key = frozenset(chosen)
        if len(common) < N_POINTS or key in seen:
            continue
        seen.add(key)
        order = tuple(sorted(chosen, key=lambda i: scene["views"][i]["name"]))
        windows.append((order, rng.choice(np.array(sorted(common)), size=N_POINTS, replace=False)))
    return windows


def window_jobs(scene: dict, windows: list, meta: dict, base_seed: int) -> list[dict]:
    jobs = []
    for k, (order, point_ids) in enumerate(windows):
        views = [scene["views"][i] for i in order]
        jobs.append({**meta, "Window": k, "seed": base_seed + k, "names": [v["name"] for v in views],
                     "cameras": [v["P"] for v in views],
                     "world": np.stack([scene["points"][p] for p in point_ids]),
                     "tracks": np.stack([[v["obs"][p] for p in point_ids] for v in views])})
    return jobs


def score_window(job: dict) -> list[dict]:
    """All variants of one window: unperturbed and each injection family."""
    import mv_window as mw
    from synthetic_data.injection import make_window

    rows = []
    rigid = make_window(np.random.default_rng(0), "rigid", 0.0, 0.0, cameras=job["cameras"], world=job["world"])["clean"]
    variants = [("none", 0.0)] + list(INJECTIONS.items())
    for v, (family, dose) in enumerate(variants):
        tracks, induced = job["tracks"], 0.0
        if family != "none":
            rng = np.random.default_rng(job["seed"] * 10 + v)
            injected = make_window(rng, family, dose, 0.0, cameras=job["cameras"], world=job["world"])
            tracks, induced = job["tracks"] + (injected["clean"] - rigid), injected["induced_px"]
        stats = mw.window_statistics(tracks, 1.0, job["seed"], with_legacy=False)
        stats.pop("_rec", None)
        reprojection = float(np.sqrt(np.mean(np.sum((job["tracks"] - rigid) ** 2, axis=2))))
        rows.append({key: job[key] for key in ("Dataset", "Scene", "Family", "Role", "Window", "seed")}
                    | {"Views": ";".join(job["names"]), "Injection": family, "Dose": dose,
                       "Induced_px": induced, "Reprojection_RMS_px": reprojection} | stats)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--recon_root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=MAX_WINDOWS_PER_SCENE)
    parser.add_argument("--results_name", default=None)
    parser.add_argument("--count_only", action="store_true",
                        help="only count the windows found per scene (geometry, no statistic computed)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"exploration: window seeds + {SMOKE_OFFSET:,}, smoke_* output files")
    args = parser.parse_args()
    seed0 = WINDOW_SEED + (SMOKE_OFFSET if args.smoke else 0)

    if args.split:
        batches = pd.read_csv(RESULTS_DIR / "real_reconstruction_batches.csv", comment="#")
        split = draw_split(batches)
        save_results(split, "real_pencil_split.csv", SPLIT_SEED, "Calibration / test scenes for real_pencil")
        print(split.to_string())
        return

    assert args.batch is not None, "give --split or --batch"
    split = pd.read_csv(RESULTS_DIR / "real_pencil_split.csv", comment="#")
    order = split.sort_values(["Dataset", "Scene"]).reset_index(drop=True)
    jobs, coverage = [], []
    for index, row in order.iterrows():
        if row["Batch"] != args.batch:
            continue
        model = (args.recon_root / f"benchmark_real_batch{args.batch}" / row["Dataset"] / row["Scene"]
                 / SOURCE_PIPELINE / f"seed_{SOURCE_SEED}" / "sparse" / "0")
        windows = []
        if model.exists():
            scene = load_scene(model)
            rng = np.random.default_rng(seed0 + 1000 * index)
            windows = find_windows(scene, rng, args.max_windows)
            meta = {k: row[k] for k in ("Dataset", "Scene", "Family", "Role")}
            jobs += window_jobs(scene, windows, meta, seed0 + 1000 * index)
        coverage.append({"Dataset": row["Dataset"], "Scene": row["Scene"], "Role": row["Role"],
                         "Model_Found": model.exists(), "N_Windows": len(windows)})
        print(f"{row['Dataset']}/{row['Scene']} ({row['Role']}): {len(windows)} windows", flush=True)
    if args.count_only:
        counts = pd.DataFrame(coverage)
        print(counts.groupby("Role")["N_Windows"].sum().to_string())
        return

    # one BLAS thread per worker process: with several workers, multi-threaded numpy
    # oversubscribes the CPU (exploration: 0.5 s per window single-threaded, minutes otherwise)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for window_rows in pool.map(score_window, jobs) for r in window_rows]
    name = args.results_name or f"{'smoke_' if args.smoke else ''}real_pencil_windows_batch{args.batch}.csv"
    save_results(pd.DataFrame(rows), name, seed0, f"Real-data pencil test windows, batch {args.batch}")
    save_results(pd.DataFrame(coverage), name.replace("windows", "coverage"), seed0,
                 f"Real-data pencil test: windows found per scene, batch {args.batch}")


if __name__ == "__main__":
    main()
