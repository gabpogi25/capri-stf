"""
experiments/eval/noise_model.py

noise-model check (docs/prereg/evaluation_plan.md): does the drift-test evidence of
pencil drift experiment depend on its noise model? pencil drift experiment perturbs points isotropically in a unit
chart. Here each measurement is TRIANGULATED from noisy pixels.

Per scene (paired arms): 12 points uniform in [-1, 1]^3, six arc cameras
(synthetic_data.injection). Measurement A = linear triangulation from views
{0, 2, 4}, B from views {1, 3, 5} (true cameras, normalized pixels, sigma =
0.5 px). rho = RMS 3D error of A and B (world units). Label 1: B's world points
are displaced uniformly in a ball of radius f rho, f ~ U(1.5, 5); label 0: no
displacement. Arm "triangulated": as above. Arm "isotropic": the same scene and
displacement, but each measurement is the true point plus isotropic Gaussian 3D
noise with the same per-scene RMS rho. Both arms then get an independent random
projective frame and per-point scale and sign. Scores (lab.py, as pencil drift experiment) are
divided by rho.

Material change (pre-registered): |AUC_tri - AUC_iso| > 0.02 or |TPR_tri - TPR_iso|
> 0.05 for pencil_q10 with the paired 95% CI excluding 0, or the pencil-minus-
registration AUC difference has CIs of opposite sign in the two arms.
Output: noise_model.csv, noise_model_summary.csv. Seed base 31,000,000.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from experiment_utils import save_results  # noqa: E402
from algorithms.consistency.pencil import pencil_drift_score  # noqa: E402
from synthetic_data.injection import HEIGHT, WIDTH, arc_cameras, project  # noqa: E402
import lab  # noqa: E402

SEED = 31_000_000
N_SCENES, N_POINTS, NOISE_PX, N_BOOT = 2000, 12, 0.5, 2000
HALF = max(WIDTH, HEIGHT) / 2.0
NORMALIZER = np.array([[1 / HALF, 0, -WIDTH / 2 / HALF], [0, 1 / HALF, -HEIGHT / 2 / HALF], [0, 0, 1.0]])
METHODS = {"pencil_q10": lambda s: pencil_drift_score(s["A"], s["B"], s["delta"]),
           "registration_rms": lab.b_drift_registration, "median_tuple_dJ": lab.b_drift_median_tuples}


def triangulate(cameras: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    """Linear triangulation in normalized pixels; (N, 3) world points."""
    cams = [NORMALIZER @ P for P in cameras]
    pts = [(np.c_[x, np.ones(len(x))] @ NORMALIZER.T)[:, :2] for x in pixels]
    out = []
    for i in range(len(pts[0])):
        rows = [r for P, x in zip(cams, pts) for r in (x[i, 0] * P[2] - P[0], x[i, 1] * P[2] - P[1])]
        X = np.linalg.svd(np.array(rows))[2][-1]
        out.append(X[:3] / X[3])
    return np.array(out)


def scene(index: int) -> list[dict]:
    rng = np.random.default_rng(SEED + index)
    cameras = arc_cameras(6)
    world = rng.uniform(-1, 1, (N_POINTS, 3))
    label = index % 2
    f = rng.uniform(1.5, 5.0)
    direction = rng.standard_normal((N_POINTS, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.uniform(size=(N_POINTS, 1)) ** (1 / 3)
    measured = {}
    for name, views, w in (("A", (0, 2, 4), world), ("B", (1, 3, 5), world)):
        pixels = [project(cameras[v], w) + rng.normal(0, NOISE_PX, (N_POINTS, 2)) for v in views]
        measured[name] = triangulate([cameras[v] for v in views], pixels)
    rho = float(np.sqrt(np.mean(np.concatenate([measured[k] - world for k in "AB"]) ** 2) * 3))
    shift = label * f * rho * radius * direction
    iso = {k: world + rng.normal(0, rho / np.sqrt(3), (N_POINTS, 3)) for k in "AB"}
    rows = []
    for arm, points in (("triangulated", measured), ("isotropic", iso)):
        A, B = points["A"], points["B"] + shift
        sample = {"A": lab.gauge(np.vstack([A.T, np.ones(N_POINTS)]), rng),
                  "B": lab.gauge(np.vstack([B.T, np.ones(N_POINTS)]), rng), "delta": rho}
        rows.append({"Scene": index, "Seed": SEED + index, "Arm": arm, "Label_Excessive": label, "f": f,
                     "rho": rho, **{m: fn(sample) for m, fn in METHODS.items()}})
    return rows


def auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def tpr5(y, s):
    return float(np.mean(s[y == 1] > np.quantile(s[y == 0], 0.95)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 9_000_000)
    wide = {arm: df[df.Arm == arm].set_index("Scene").sort_index() for arm in ("triangulated", "isotropic")}
    y = wide["triangulated"]["Label_Excessive"].to_numpy()
    n = len(y)
    boots = [rng.integers(0, n, n) for _ in range(N_BOOT)]

    def stat(fn, arm, m, idx=None):
        s = wide[arm][m].to_numpy()
        return fn(y if idx is None else y[idx], s if idx is None else s[idx])

    rows = []
    for m in METHODS:
        for name, fn in (("AUC", auc), ("TPR_at_5pct", tpr5)):
            est = {a: stat(fn, a, m) for a in wide}
            draws = np.array([[stat(fn, a, m, b) for a in wide] for b in boots])
            diff = draws[:, 0] - draws[:, 1]
            rows.append({"Method": m, "Metric": name, "Triangulated": est["triangulated"], "Isotropic": est["isotropic"],
                         "Diff_tri_minus_iso": est["triangulated"] - est["isotropic"],
                         "Diff_CI_Lo": np.quantile(diff, 0.025), "Diff_CI_Hi": np.quantile(diff, 0.975)})
    for arm in wide:
        draws = np.array([stat(auc, arm, "pencil_q10", b) - stat(auc, arm, "registration_rms", b) for b in boots])
        rows.append({"Method": "pencil_q10 - registration_rms", "Metric": f"AUC diff ({arm})",
                     "Triangulated" if arm == "triangulated" else "Isotropic":
                     stat(auc, arm, "pencil_q10") - stat(auc, arm, "registration_rms"),
                     "Diff_CI_Lo": np.quantile(draws, 0.025), "Diff_CI_Hi": np.quantile(draws, 0.975)})
    out = pd.DataFrame(rows)
    p = out[(out.Method == "pencil_q10") & (out.Metric.isin(["AUC", "TPR_at_5pct"]))]
    material = any((abs(r.Diff_tri_minus_iso) > (0.02 if r.Metric == "AUC" else 0.05)) and
                   (r.Diff_CI_Lo > 0 or r.Diff_CI_Hi < 0) for r in p.itertuples())
    signs = out[out.Method == "pencil_q10 - registration_rms"]
    flip = (signs.Diff_CI_Lo.max() > 0) and (signs.Diff_CI_Hi.min() < 0)
    out["Material_change"] = bool(material or flip)
    return out


def main() -> None:
    df = pd.DataFrame([row for i in range(N_SCENES) for row in scene(i)])
    summary = summarize(df)
    print(summary.to_string(index=False))
    save_results(df, "noise_model.csv", SEED, "noise-model check: drift test, triangulated vs isotropic noise (paired)")
    save_results(summary, "noise_model_summary.csv", SEED, "noise-model check: summary and material-change verdict")


if __name__ == "__main__":
    main()
