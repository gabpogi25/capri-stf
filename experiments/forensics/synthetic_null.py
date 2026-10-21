"""
experiments/forensics/synthetic_null.py

Stage S1: is the single-image statistic calibrated on
data that follow its own noise model, and does it respond to inconsistency?

Part 1, synthetic segments: cross-fitted vanishing-point z-scores on segments
drawn exactly from the noise model (target: KS distance to N(0,1) <= 0.02 at
N >= 20,000). Also the response to an injected inconsistency (extra random
rotation of every segment, 0.05-0.5 degrees).

Part 2, rendered perspective scenes: the full image pipeline (LSD + VP test +
bend test) on renderings with radial distortion k1 in {0, 0.05, 0.15}. This
shows the statistic's response, AND that its scale on rendered images differs
from the model (sigma_p is not calibrated for renderings), which is why the real-media null gate
uses an empirical null on authentic photographs.

Neither part says anything about generated media.
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import save_csv
from forensics.config import load_config
from forensics.pipeline_image import analyze_gray
from forensics.structure.synthetic import render_perspective_scene, synthetic_segments
from forensics.structure.vp_crossfit import cross_fitted_vp_test

SEED = 42201


def main() -> None:
    config = load_config(overrides={"vanishing_points": {"ransac_iters": 500}})
    rng = np.random.default_rng(SEED)

    rows = []
    for inconsistency in (0.0, 0.05, 0.1, 0.2, 0.5):
        n_images = 170 if inconsistency == 0.0 else 30
        z = []
        for _ in range(n_images):
            z += [r["z"] for r in cross_fitted_vp_test(synthetic_segments(rng, inconsistency_deg=inconsistency),
                                                       config, rng)]
        z = np.asarray(z)
        ks = stats.kstest(z, "norm")
        rows.append({"Part": "synthetic_segments", "Inconsistency_deg": inconsistency, "N_z": len(z),
                     "Mean_z": z.mean(), "SD_z": z.std(), "KS_vs_N01": ks.statistic, "KS_p": ks.pvalue,
                     "Frac_abs_z_gt_1.96": float(np.mean(np.abs(z) > 1.96))})
        print(rows[-1])
    segments = pd.DataFrame(rows)

    rendered = []
    for k1 in (0.0, 0.05, 0.15):
        for scene in range(10):
            scene_rng = np.random.default_rng(SEED + 1000 + scene)
            gray = render_perspective_scene(scene_rng, distortion_k1=k1)
            evidence = analyze_gray(gray, config, scene_rng)
            by_kind = {}
            for c in evidence.constraints:
                by_kind.setdefault(c.kind, []).append(c.z)
            rendered.append({"Part": "rendered_scene", "Distortion_k1": k1, "Scene": scene,
                             "N_Segments": evidence.n_segments, "N_Constraints": evidence.n_constraints,
                             "Mean_z2": evidence.statistic_sum / max(evidence.dof, 1),
                             "SD_z_vp": float(np.std(by_kind.get("vp_angle", [np.nan]))),
                             "SD_z_bend": float(np.std(by_kind.get("bend", [np.nan]))),
                             "Decision_uncalibrated": evidence.decision})
        print(pd.DataFrame(rendered).groupby("Distortion_k1")[["Mean_z2", "SD_z_vp", "SD_z_bend"]].median())
    save_csv(segments, "synthetic_segments.csv", SEED, "S1: VP z-score calibration on model-consistent segments", config)
    save_csv(pd.DataFrame(rendered), "rendered_scenes.csv", SEED, "S1: image pipeline on rendered scenes", config)


if __name__ == "__main__":
    main()
