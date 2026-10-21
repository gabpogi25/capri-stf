"""
experiments/run_abstract_photo_demo.py

Illustration for the graphical abstracts on real photos (not an evaluation).

1. COLMAP (pycolmap) structure from motion on ten public photos of the Sacre-Coeur
   (figures/abstracts/photos/sacre_coeur/, from the hloc sample data): SIFT features,
   exhaustive matching, incremental mapping. This gives cameras and a sparse model.
2. The photos are split into two disjoint view subsets V_A and V_B. Every model point
   seen at least twice in each subset is triangulated twice, once from its V_A
   observations and once from its V_B observations (linear triangulation with the
   SfM cameras), and each result is put in its own random projective frame with
   random point scales. These are two measurements of the same points.
3. The pencil statistic Q_0.1 compares them. In the injected version, 30% of the
   points get 8 px of jitter on their V_B observations before triangulation
   (the point-jitter injection of the evaluation, on real detections).

The cameras come from one joint reconstruction, so A and B share calibration; the
measurements (2D detections) of the two subsets are disjoint. Writes
experiments/results/abstract_photo_*.csv via save_results.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.consistency.pencil import pencil_discrepancies, pencil_drift_score, pencil_point_scores  # noqa: E402
from experiments.experiment_utils import get_experiment_seed, save_results  # noqa: E402

EXPERIMENT_ID = 22
PHOTOS = ROOT / "figures" / "abstracts" / "photos" / "sacre_coeur"
SUBSET_A = ["02928139_3448003521.jpg", "10265353_3838484249.jpg", "44120379_8371960244.jpg",
            "17295357_9106075285.jpg", "71295362_4051449754.jpg"]
INJECTED_FRACTION = 0.3
JITTER_PX = 8.0


def structure_from_motion(work: Path, seed: int) -> pycolmap.Reconstruction:
    """Single-threaded and seeded at every stage, so that reruns give the same model."""
    images = work / "images"
    shutil.copytree(PHOTOS, images)
    db = work / "database.db"
    names = sorted(p.name for p in images.glob("*.jpg"))
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = 1
    pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              extraction_options=extraction, device=pycolmap.Device.cpu)
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = 1
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = seed
    verification.ransac.num_threads = 1
    pycolmap.match_exhaustive(db, matching_options=matching, verification_options=verification,
                              device=pycolmap.Device.cpu)
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = seed
    options.num_threads = 1
    models = pycolmap.incremental_mapping(db, images, work / "sparse", options=options)
    return max(models.values(), key=lambda r: (r.num_reg_images(), r.num_points3D()))


def camera_matrix(image) -> np.ndarray:
    pose = image.cam_from_world
    pose = pose() if callable(pose) else pose
    return pose.matrix()                                   # [R | t]; observations are normalised below


def triangulate(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Linear (DLT) triangulation from (camera [R|t], normalised point) pairs."""
    rows = []
    for P, x in observations:
        rows.append(x[0] * P[2] - P[0])
        rows.append(x[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    return vt[-1]


def random_gauge(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H = np.eye(4) + 0.3 * rng.standard_normal((4, 4))
    scales = rng.uniform(0.5, 2.0, points.shape[1]) * rng.choice([-1.0, 1.0], points.shape[1])
    return (H @ points) * scales


def main() -> None:
    seed = get_experiment_seed(EXPERIMENT_ID)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        rec = structure_from_motion(Path(tmp), seed)
    images = {im.image_id: im for im in rec.images.values()}
    subset = {iid: ("A" if im.name in SUBSET_A else "B") for iid, im in images.items()}

    common, obs_a, obs_b, pix = [], [], [], []
    for pid, point in rec.points3D.items():
        per = {"A": [], "B": []}
        for el in point.track.elements:
            im = images[el.image_id]
            xy = im.points2D[el.point2D_idx].xy
            per[subset[el.image_id]].append((el.image_id, np.asarray(xy, dtype=float)))
        if len(per["A"]) >= 2 and len(per["B"]) >= 2:
            common.append(pid)
            obs_a.append(per["A"])
            obs_b.append(per["B"])

    n = len(common)
    injected = np.zeros(n, dtype=bool)
    injected[rng.choice(n, int(round(INJECTED_FRACTION * n)), replace=False)] = True

    def measure(observations, jitter: np.ndarray) -> np.ndarray:
        X = []
        for k, obs in enumerate(observations):
            pairs = []
            for iid, xy in obs:
                im = images[iid]
                xy = xy + jitter[k] * rng.standard_normal(2)
                normalised = im.camera.cam_from_img(xy[None, :])[0]
                pairs.append((camera_matrix(im), normalised))
            X.append(triangulate(pairs))
        X = np.array(X).T
        return X / np.linalg.norm(X, axis=0, keepdims=True)

    no_jitter = np.zeros(n)
    A = random_gauge(measure(obs_a, no_jitter), rng)
    B_consistent = random_gauge(measure(obs_b, no_jitter), rng)
    B_injected = random_gauge(measure(obs_b, np.where(injected, JITTER_PX, 0.0)), rng)

    S_consistent = pencil_drift_score(A, B_consistent, seed=seed)
    S_injected = pencil_drift_score(A, B_injected, seed=seed)
    scores = pencil_point_scores(A, B_injected, seed=seed)
    D_consistent = pencil_discrepancies(A, B_consistent, seed=seed)
    D_injected = pencil_discrepancies(A, B_injected, seed=seed)

    # Model points and cameras for the renders (Euclidean SfM frame, display only).
    model = pd.DataFrame([{"Point": pid, "X": p.xyz[0], "Y": p.xyz[1], "Z": p.xyz[2], "R": p.color[0], "G": p.color[1],
                           "B": p.color[2], "Track": p.track.length(), "Common": pid in set(common)}
                          for pid, p in rec.points3D.items()])
    cams = []
    for iid, im in images.items():
        pose = camera_matrix(im)
        R, t = pose[:, :3], pose[:, 3]
        c = -R.T @ t
        f, cx, cy = im.camera.params[0], im.camera.params[1], im.camera.params[2]
        cams.append({"Image": im.name, "Subset": subset[iid], "Cx": c[0], "Cy": c[1], "Cz": c[2],
                     **{f"R{r}{k}": R[r, k] for r in range(3) for k in range(3)},
                     "Focal": f, "Px": cx, "Py": cy, "Width": im.camera.width, "Height": im.camera.height})
    cams = pd.DataFrame(cams)

    # Observations of the common points in every image (for photo overlays).
    index = {pid: k for k, pid in enumerate(common)}
    overlay = []
    for k, pid in enumerate(common):
        for el in rec.points3D[pid].track.elements:
            xy = images[el.image_id].points2D[el.point2D_idx].xy
            overlay.append({"Point": pid, "Image": images[el.image_id].name, "Subset": subset[el.image_id],
                            "x": xy[0], "y": xy[1], "Injected": bool(injected[k]), "Pencil_Score": scores[k]})
    overlay = pd.DataFrame(overlay)

    summary = pd.DataFrame([
        {"Quantity": "Photos registered", "Value": rec.num_reg_images()},
        {"Quantity": "Model points", "Value": rec.num_points3D()},
        {"Quantity": "Mean reprojection error (px)", "Value": rec.compute_mean_reprojection_error()},
        {"Quantity": "Points seen twice in each subset", "Value": n},
        {"Quantity": "Injected points (8 px jitter on V_B)", "Value": int(injected.sum())},
        {"Quantity": "Pencil cross-ratios compared", "Value": int(np.isfinite(D_consistent).sum())},
        {"Quantity": "S = Q0.1, consistent", "Value": S_consistent},
        {"Quantity": "S = Q0.1, injected", "Value": S_injected},
        {"Quantity": "Median point score, injected points", "Value": float(np.nanmedian(scores[injected]))},
        {"Quantity": "Median point score, other points", "Value": float(np.nanmedian(scores[~injected]))},
    ])
    dist = pd.DataFrame({
        "Version": np.repeat(["consistent", "injected"], [np.isfinite(D_consistent).sum(), np.isfinite(D_injected).sum()]),
        "D": np.concatenate([D_consistent[np.isfinite(D_consistent)], D_injected[np.isfinite(D_injected)]]),
    })
    name = "Abstract photo demo: COLMAP on 10 Sacre-Coeur photos, two view subsets, injected jitter (illustration)"
    save_results(model, "abstract_photo_model.csv", seed, name)
    save_results(cams, "abstract_photo_cameras.csv", seed, name)
    save_results(overlay, "abstract_photo_observations.csv", seed, name)
    save_results(summary, "abstract_photo_summary.csv", seed, name)
    save_results(dist, "abstract_photo_discrepancies.csv", seed, name)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
