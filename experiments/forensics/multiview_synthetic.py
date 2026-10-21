"""
experiments/forensics/multiview_synthetic.py

Stage S6a: checks of the multi-view Delta_J branch on
a synthetic rigid scene (60 points, 6 cameras on a 30-degree arc, f = 800 px,
1280 x 720 px). Subset A = views {1,3,5}, subset B = views {2,4,6}.

Primary statistic: the pencil drift statistic Q_0.1 of |d log|CR|| between the
two subset reconstructions (docs/MATHEMATICS.md, Section 4; the video window
statistic). The Delta_J statistic on gated 6-tuples is reported in secondary
(legacy) columns.

  (i)   noise-free: the statistic Q_0.1 is numerically zero (Lemma 2.1: the two
        reconstructions differ by a change of frame). The bundle-adjustment
        stopping tolerance (~1e-7) sets the floor; single pencils through a
        nearly zero bracket amplify it by 1/|bracket|, so the maximum over
        all pencils is reported separately and is not the test;
  (ii)  noise scaling: the median standardized |Delta_J| should grow linearly
        with pixel noise (first-order law; log-log slope ~ 1);
  (iii) injection: one in four points displaced in subset B's views only;
        the statistic should increase with the displacement.
These validate the implementation, not the forensic hypothesis.
"""

import numpy as np
import pandas as pd

from common import save_csv
from forensics.config import load_config
from algorithms.consistency.pencil import pencil_discrepancies
from forensics.video.multiview import delta_j, projective_reconstruction, sample_gated_tuples
from forensics.video.synthetic import normalize_pixels, synthetic_rigid_views

SEED = 42206
N_SCENES = 10


def window(seed: int, noise_px: float, offset: float, sigma4_min: float) -> dict:
    rng = np.random.default_rng(seed)
    displaced = {"indices": list(range(0, 60, 4)), "views": [1, 3, 5], "offset": offset} if offset > 0 else None
    tracks, _ = synthetic_rigid_views(rng, noise_px=noise_px, displaced=displaced)
    views = normalize_pixels(tracks)
    rec_a = projective_reconstruction([views[0], views[2], views[4]], seed)
    rec_b = projective_reconstruction([views[1], views[3], views[5]], seed)
    D = pencil_discrepancies(rec_a["X"], rec_b["X"], seed=seed)
    D = D[np.isfinite(D)]
    tuples = sample_gated_tuples(rec_a["X"], rec_b["X"], 200, rng, sigma4_min)
    result = delta_j(rec_a["X"], rec_b["X"], tuples)
    return {"Pencil_Q10": float(np.quantile(D, 0.1)), "Pencil_Max": float(D.max()), "N_Pencils": int(D.size),
            "N_Tuples": len(tuples), "Max_Abs_DeltaJ": float(np.abs(result["delta"]).max()),
            "Median_Standardized": float(np.median(result["standardized"])),
            "RMS_A": rec_a["rms"], "RMS_B": rec_b["rms"]}


def main() -> None:
    config = load_config()
    sigma4_min = config["video"]["sigma4_min"]
    rows = []
    for scene in range(N_SCENES):
        rows.append({"Check": "i_noise_free", "Noise_px": 0.0, "Offset": 0.0, "Scene": scene,
                     **window(SEED + scene, 0.0, 0.0, sigma4_min)})
        for noise in (0.25, 0.5, 1.0, 2.0):
            rows.append({"Check": "ii_noise_scaling", "Noise_px": noise, "Offset": 0.0, "Scene": scene,
                         **window(SEED + scene, noise, 0.0, sigma4_min)})
        for offset in (0.01, 0.03, 0.1, 0.3):
            rows.append({"Check": "iii_injection", "Noise_px": 0.5, "Offset": offset, "Scene": scene,
                         **window(SEED + scene, 0.5, offset, sigma4_min)})
    df = pd.DataFrame(rows)
    noise_free = df[df.Check == "i_noise_free"]
    print(f"(i) noise-free: max pencil Q0.1 {noise_free['Pencil_Q10'].max():.2e}, "
          f"max |Delta_J| {noise_free['Max_Abs_DeltaJ'].max():.2e} (target < 1e-6, BA tolerance); "
          f"max single pencil {noise_free['Pencil_Max'].max():.2e} (near-zero brackets, not the test)")
    for column in ("Pencil_Q10", "Median_Standardized"):
        scaling = df[df.Check == "ii_noise_scaling"].groupby("Noise_px")[column].median()
        slope = np.polyfit(np.log(scaling.index), np.log(scaling.values), 1)[0]
        print(f"(ii) {column} by noise: {scaling.to_dict()}; log-log slope = {slope:.3f}")
        injected = pd.concat([df[(df.Check == "ii_noise_scaling") & (df.Noise_px == 0.5)].assign(Offset=0.0),
                              df[df.Check == "iii_injection"]]).groupby("Offset")[column].median()
        print(f"(iii) {column} by offset: {injected.to_dict()}")
    save_csv(df, "multiview_synthetic.csv", SEED, "S6a: Delta_J checks on a synthetic rigid scene", config)


if __name__ == "__main__":
    main()
