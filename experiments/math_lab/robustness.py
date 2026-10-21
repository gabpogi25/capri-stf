"""
experiments/math_lab/robustness.py

Distribution-shift check for the selected candidates: the hidden test uses the
same generator as dev, so it cannot reveal methods that rely on generator
details. Here the data model is changed in ways the candidates were NOT told:

  drift_gauss      points ~ N(0, I) in 3D instead of uniform in a cube
  nonrigid_gauss   points ~ N(0, I) instead of a cube; same clustered motion
  nonrigid_random  moved set is a RANDOM subset (not a spatial cluster)

Usage: for m in baseline_registration baseline_ransac d3 d4 n1 n4; do
         python experiments/math_lab/robustness.py --only $m; done
"""

import argparse
import json

import numpy as np

import lab
from synthetic_data.generator import measure_with_gaussian_noise, perturb_within_ball, unit_representatives


def gaussian_points(n, rng):
    return np.vstack([rng.standard_normal((3, n)), np.ones((1, n))])


def gen_drift_gauss(rng):
    points = gaussian_points(12, rng)
    delta = float(np.exp(rng.uniform(np.log(0.003), np.log(0.1))))
    label = int(rng.random() < 0.5)
    radius = delta * (rng.uniform(1.5, 5.0) if label else 1.0)
    return {"A": lab.gauge(perturb_within_ball(points, radius, rng), rng),
            "B": lab.gauge(perturb_within_ball(points, radius, rng), rng), "delta": delta}, label


def gen_nonrigid(rng, gaussian: bool, clustered: bool, n: int = 20):
    points = gaussian_points(n, rng) if gaussian else lab.generate_generic_configuration(n, scale=1.0, rng=rng)
    q = unit_representatives(points)
    m = int(rng.integers(int(0.2 * n), int(0.4 * n) + 1))
    if clustered:
        centre = q[:, int(rng.integers(n))]
        moved = np.argsort(np.linalg.norm(q - centre[:, None], axis=0))[:m]
    else:
        moved = rng.choice(n, m, replace=False)
    amplitude = float(np.exp(rng.uniform(np.log(0.003), np.log(0.05))))
    direction = rng.standard_normal(4); direction /= np.linalg.norm(direction)
    A = measure_with_gaussian_noise(points, 1e-3, rng)
    B = measure_with_gaussian_noise(points, 1e-3, rng)
    B[:, moved] += amplitude * np.outer(direction, rng.uniform(0.5, 1.0, m))
    labels = np.zeros(n, int); labels[moved] = 1
    return {"A": lab.gauge(A, rng), "B": lab.gauge(B, rng)}, labels


_LOADED = {}


def _load(c):
    if c not in _LOADED:
        _LOADED[c] = lab.load_score_fn(str(lab.ROOT / "candidates" / c / "method.py"))
    return _LOADED[c]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=515151)
    parser.add_argument("--only", help="run a single method (candidates must run in separate processes: "
                                       "their helper modules can share names such as chart.py)")
    args = parser.parse_args()
    cand = lambda c: (lambda s, _c=c: _load(_c)(s))  # noqa: E731  (lazy: only the selected candidate is imported)
    suites = {
        "drift_gauss": ("drift", gen_drift_gauss,
                        {"baseline_registration": lab.b_drift_registration, "d3": cand("d3"), "d4": cand("d4")}),
        "nonrigid_gauss": ("nonrigid", lambda r: gen_nonrigid(r, True, True),
                           {"baseline_ransac": lab.b_outlier_ransac_registration, "n1": cand("n1"), "n4": cand("n4")}),
        "nonrigid_random": ("nonrigid", lambda r: gen_nonrigid(r, False, False),
                            {"baseline_ransac": lab.b_outlier_ransac_registration, "n1": cand("n1"), "n4": cand("n4")}),
    }
    for k, (name, (track, gen, methods)) in enumerate(suites.items()):
        rng = np.random.default_rng(args.seed + k)
        data = [gen(rng) for _ in range(args.n)]
        for method, fn in methods.items():
            if args.only and method != args.only:
                continue
            print(json.dumps({"suite": name, "method": method, **lab.evaluate(track, fn, data)}), flush=True)


if __name__ == "__main__":
    main()
