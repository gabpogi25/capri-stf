"""
experiments/run_figure_examples.py

Data for the illustrative paper figures (figures/paper/; not an evaluation).

1. Design figure: one simulated window of the evaluation (synthetic_data/injection.py):
   40 points, six cameras on a 30-degree arc, and each injection family applied to the same
   points at a display dose of 8 px (swaps: 15% of the points). For one view of subset B (view 5
   for drift, view 3 otherwise) the file records each point's pixel position without and with the injection.
2. Teaser figure: the Sacre-Coeur photo demo (experiments/run_abstract_photo_demo.py). Each
   shared point is triangulated once from its V_A detections and once from its V_B detections
   with the SfM cameras, and each set is put in its own projective frame (a display gauge), to
   show that the same points have different coordinates in the two reconstructions.

Seed 90,500,000 (exploration band). Output: figure_design_window.csv, figure_design_scene.csv,
figure_teaser_frames.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402
from synthetic_data.injection import arc_cameras, make_window, project, scene_points  # noqa: E402

SEED = 90_500_000
SHOW_VIEW = {"drift": 5}                         # views of subset B = {1, 3, 5}: drift is largest at the arc end,
DEFAULT_VIEW = 3                                 # warp in the middle; swaps act on views 3-5
FAMILIES = (("jitter", 8.0), ("drift", 8.0), ("warp", 8.0), ("swap", 0.15), ("global_projective", 8.0))


def design() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    cameras = arc_cameras(6)
    world = scene_points(rng, 40)
    rows = []
    for family, dose in FAMILIES:
        view = SHOW_VIEW.get(family, DEFAULT_VIEW)
        base = project(cameras[view], world)
        window = make_window(np.random.default_rng(SEED + len(rows) + 1), family, dose, 0.0, cameras=cameras, world=world)
        shown = window["clean"][view]
        for k in range(len(world)):
            rows.append({"Family": family, "View": view, "Point": k, "X0": base[k, 0], "Y0": base[k, 1], "X1": shown[k, 0],
                         "Y1": shown[k, 1], "Moved": bool(window["moved"][k]), "Induced_px": window["induced_px"]})
    scene = [{"Kind": "point", "Index": k, "X": p[0], "Y": p[1], "Z": p[2], "Subset": ""} for k, p in enumerate(world)]
    for v, P in enumerate(cameras):
        centre = np.linalg.svd(P)[2][-1]
        centre = centre[:3] / centre[3]
        scene.append({"Kind": "camera", "Index": v, "X": centre[0], "Y": centre[1], "Z": centre[2],
                      "Subset": "A" if v % 2 == 0 else "B"})
    return pd.DataFrame(rows), pd.DataFrame(scene)


def pixel_camera(row: pd.Series) -> np.ndarray:
    R = row[[f"R{i}{j}" for i in range(3) for j in range(3)]].to_numpy(float).reshape(3, 3)
    C = row[["Cx", "Cy", "Cz"]].to_numpy(float)
    K = np.array([[row["Focal"], 0, row["Px"]], [0, row["Focal"], row["Py"]], [0, 0, 1.0]])
    return K @ np.hstack([R, (-R @ C)[:, None]])


def triangulate(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = []
    for P, x in pairs:
        rows += [x[0] * P[2] - P[0], x[1] * P[2] - P[1]]
    return np.linalg.svd(np.array(rows))[2][-1]


def teaser() -> pd.DataFrame:
    cams = pd.read_csv(RESULTS_DIR / "abstract_photo_cameras.csv", comment="#").set_index("Image")
    obs = pd.read_csv(RESULTS_DIR / "abstract_photo_observations.csv", comment="#")
    P = {img: pixel_camera(row) for img, row in cams.iterrows()}
    rng = np.random.default_rng(SEED + 100)
    gauges = {"A": np.eye(4), "B": np.eye(4) + 0.35 * rng.standard_normal((4, 4))}
    gauges["B"][3, :3] *= 0.1        # display gauge for B (mild perspective row: no point crosses the plane at infinity)
    rows = []
    for point, group in obs.groupby("Point"):
        for subset in "AB":
            g = group[group["Subset"] == subset]
            X = gauges[subset] @ triangulate([(P[r.Image], np.array([r.x, r.y])) for r in g.itertuples()])
            rows.append({"Point": point, "Subset": subset, "X": X[0] / X[3], "Y": X[1] / X[3], "Z": X[2] / X[3],
                         "Injected": bool(g["Injected"].iloc[0]), "Pencil_Score": float(g["Pencil_Score"].iloc[0])})
    return pd.DataFrame(rows)


def main() -> None:
    window, scene = design()
    save_results(window, "figure_design_window.csv", SEED, "Illustration: one window per injection family (display dose)")
    save_results(scene, "figure_design_scene.csv", SEED, "Illustration: scene and cameras of the design figure")
    save_results(teaser(), "figure_teaser_frames.csv", SEED + 100,
                 "Illustration: shared photo points triangulated per subset; B in a display projective frame")


if __name__ == "__main__":
    main()
