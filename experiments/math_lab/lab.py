"""
experiments/math_lab/lab.py

Benchmark harness for the math-optimization round. Four tracks, each with a
fixed DEV set (seed below, visible to candidates) and a HIDDEN TEST set whose
seed is chosen only after candidates are frozen. Candidates are scored by the
same code on both.

Every track is frame-free: the two measurements of a scene are put in
independent random projective frames with random per-point scale and sign
(gauge), because real reconstructions share no coordinate chart. Only
projective invariants (or frame-fitting) can carry information.

Tracks
------
drift      scene-level. 12 points, two measurements each moved within a ball
           of radius r in the unit chart, then gauged. Label 1 if r is
           "excessive" (r = f * delta, f ~ U(1.5, 5)), 0 if r = delta.
           Candidate gets A, B (4 x 12) and delta. Score: higher = excessive.
           Metrics: AUC, TPR at 5% FPR.
cond       tuple-level. 6 points (generic and near-degenerate mixes) in a
           random frame, unit columns. Target: median over 300 draws of
           max_k |log J_k(P + N(0, 1e-3)) - log J_k(P)| (noise isotropic in the
           GIVEN coordinates). Score should predict the target.
           Metrics: Spearman rank correlation, AUC for target > 0.05.
outlier3d  point-level. 15 points, A and B measured with N(0, 1e-3) noise,
           1-3 points of B replaced by unrelated points, then gauged.
           Score: array of 15, higher = corrupted. Metrics: pooled AUC,
           precision@k (k = true number corrupted).
nonrigid   point-level. 20 points; a spatial cluster (20-40%) is moved in B by a
           smooth displacement of amplitude 0.003-0.05 (noise 1e-3), then
           gauged. Score: array of 20. Metrics: pooled AUC, precision@k.
multiview  point-level. 40 points seen in 6 views (0.5 px noise, normalized
           coordinates); ~15% of points have ONE view's observation replaced
           by another point's observation. Score: array of 40.
           Metrics: pooled AUC, precision@k.

Usage
-----
  python experiments/math_lab/lab.py baselines --track drift
  python experiments/math_lab/lab.py eval --track drift --module experiments/math_lab/candidates/d1/method.py
  python experiments/math_lab/lab.py eval ... --split test --seed <hidden>
  python experiments/math_lab/lab.py selftest
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from synthetic_data.generator import (  # noqa: E402
    generate_generic_configuration, measure_with_gaussian_noise, perturb_within_ball, unit_representatives,
)

DEV_SEED = 777001
SIZES = {"dev": {"drift": 400, "cond": 600, "outlier3d": 200, "multiview": 60, "nonrigid": 200},
         "test": {"drift": 1000, "cond": 1500, "outlier3d": 400, "multiview": 120, "nonrigid": 400}}
TRACKS = ("drift", "cond", "outlier3d", "multiview", "nonrigid")
POINT_LEVEL = {"outlier3d", "multiview", "nonrigid"}

# J1, J2, J3 bracket quads (0-based columns of a 6-point tuple), as in forensics/geometry/invariants.py
J_QUADS = (
    (((0, 1, 2, 3), (0, 1, 4, 5)), ((0, 1, 2, 4), (0, 1, 3, 5))),
    (((0, 2, 3, 4), (1, 2, 3, 5)), ((0, 2, 3, 5), (1, 2, 3, 4))),
    (((0, 1, 4, 5), (2, 3, 4, 5)), ((0, 2, 4, 5), (1, 3, 4, 5))),
)


# ---------------------------------------------------------------------------
# shared math helpers (candidates may import these)
# ---------------------------------------------------------------------------
def random_projective(rng: np.random.Generator, spread: float = 0.6) -> np.ndarray:
    """Random 4x4 map with log-normal singular values (condition ~ exp(2 * 2 * spread))."""
    q1, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    q2, _ = np.linalg.qr(rng.standard_normal((4, 4)))
    return q1 @ np.diag(np.exp(spread * rng.standard_normal(4))) @ q2


def gauge(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent random frame + random per-point scale and sign."""
    scales = np.exp(0.5 * rng.standard_normal(P.shape[1])) * rng.choice([-1.0, 1.0], P.shape[1])
    return random_projective(rng) @ P * scales


def log_abs_J_batch(P: np.ndarray, tuples: np.ndarray) -> np.ndarray:
    """log|J1..J3| for many tuples: P (4, n), tuples (T, 6) int -> (T, 3)."""
    tuples = np.asarray(tuples, dtype=int)
    out = np.zeros((len(tuples), 3))
    for k, ((n1, n2), (d1, d2)) in enumerate(J_QUADS):
        logs = []
        for quad in (n1, n2, d1, d2):
            cols = tuples[:, list(quad)]                       # (T, 4)
            M = np.transpose(P[:, cols], (1, 0, 2))            # (T, 4, 4)
            logs.append(np.log(np.abs(np.linalg.det(M)) + 1e-300))
        out[:, k] = logs[0] + logs[1] - logs[2] - logs[3]
    return out


def all_tuples(n: int) -> np.ndarray:
    return np.array(list(combinations(range(n), 6)), dtype=int)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.nanmax(np.where(np.isfinite(s), s, -np.inf)) + 1.0)
    ranks = stats.rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.05) -> float:
    y, s = np.asarray(y, int), np.asarray(s, float)
    s = np.where(np.isfinite(s), s, np.inf)
    threshold = np.quantile(s[y == 0], 1 - fpr)
    return float(np.mean(s[y == 1] > threshold))


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------
def gen_drift(rng):
    points = generate_generic_configuration(12, scale=1.0, rng=rng)
    delta = float(np.exp(rng.uniform(np.log(0.003), np.log(0.1))))
    label = int(rng.random() < 0.5)
    radius = delta * (rng.uniform(1.5, 5.0) if label else 1.0)
    A = gauge(perturb_within_ball(points, radius, rng), rng)
    B = gauge(perturb_within_ball(points, radius, rng), rng)
    return {"A": A, "B": B, "delta": delta}, label


def _cond_config(rng):
    kind = rng.choice(["generic", "near_plane4", "near_plane5", "near_line3"])
    X = rng.uniform(-1, 1, (3, 6))
    eps = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.3))))
    if kind != "generic":
        m = {"near_plane4": 4, "near_plane5": 5, "near_line3": 3}[kind]
        idx = rng.choice(6, m, replace=False)
        if kind == "near_line3":
            a, b = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            for j, i in enumerate(idx):
                X[:, i] = a + rng.uniform(-1, 1) * b + eps * rng.standard_normal(3)
        else:
            n = rng.standard_normal(3); n /= np.linalg.norm(n)
            X[:, idx] -= np.outer(n, n @ X[:, idx])
            X[:, idx] += eps * np.outer(n, rng.standard_normal(m))
    P = np.vstack([X, np.ones(6)])
    return unit_representatives(random_projective(rng) @ P)


def gen_cond(rng, n_draws: int = 300, sigma: float = 1e-3):
    P = _cond_config(rng)
    base = log_abs_J_batch(P, np.arange(6)[None])[0]
    noise = sigma * rng.standard_normal((n_draws, 4, 6))
    perturbed = np.array([log_abs_J_batch(P + e, np.arange(6)[None])[0] for e in noise])
    target = float(np.median(np.max(np.abs(perturbed - base), axis=1)))
    return {"P": P, "sigma": sigma}, target


def gen_outlier3d(rng, n: int = 15):
    points = generate_generic_configuration(n, scale=1.0, rng=rng)
    k = int(rng.integers(1, 4))
    bad = rng.choice(n, k, replace=False)
    A = measure_with_gaussian_noise(points, 1e-3, rng)
    B = measure_with_gaussian_noise(points, 1e-3, rng)
    B[:, bad] = unit_representatives(generate_generic_configuration(6, scale=1.0, rng=rng)[:, :k])
    labels = np.zeros(n, int); labels[bad] = 1
    return {"A": gauge(A, rng), "B": gauge(B, rng)}, labels


def gen_multiview(rng, n: int = 40):
    from forensics.video.synthetic import normalize_pixels, synthetic_rigid_views

    tracks, _ = synthetic_rigid_views(rng, n_points=n, n_views=6, noise_px=0.5)
    views = normalize_pixels(tracks)
    labels = np.zeros(n, int)
    bad = rng.choice(n, max(1, int(round(0.15 * n))), replace=False)
    for i in bad:
        v = int(rng.integers(6))
        other = int(rng.choice([j for j in range(n) if j != i]))
        views[v, i] = views[v, other]
        labels[i] = 1
    return {"views": views}, labels


def gen_nonrigid(rng, n: int = 20):
    """
    20 points; a spatial cluster (the nearest 20-40% of points to a random centre)
    is displaced by a smooth field of amplitude d ~ logU(0.003, 0.05) in B only.
    Both measurements carry N(0, 1e-3) noise; both are gauged. Label = displaced.
    """
    points = generate_generic_configuration(n, scale=1.0, rng=rng)
    q = unit_representatives(points)
    m = int(rng.integers(int(0.2 * n), int(0.4 * n) + 1))
    centre = q[:, int(rng.integers(n))]
    moved = np.argsort(np.linalg.norm(q - centre[:, None], axis=0))[:m]
    amplitude = float(np.exp(rng.uniform(np.log(0.003), np.log(0.05))))
    direction = rng.standard_normal(4); direction /= np.linalg.norm(direction)
    A = measure_with_gaussian_noise(points, 1e-3, rng)
    B = measure_with_gaussian_noise(points, 1e-3, rng)
    weights = rng.uniform(0.5, 1.0, m)
    B[:, moved] += amplitude * np.outer(direction, weights)
    labels = np.zeros(n, int); labels[moved] = 1
    return {"A": gauge(A, rng), "B": gauge(B, rng)}, labels


GENERATORS = {"nonrigid": gen_nonrigid, "drift": gen_drift, "cond": gen_cond, "outlier3d": gen_outlier3d, "multiview": gen_multiview}


def dataset(track: str, split: str = "dev", seed: int | None = None):
    seed = DEV_SEED if split == "dev" else seed
    if seed is None:
        raise SystemExit("the test split needs --seed (chosen after candidates are frozen)")
    cache = ROOT / "data" / f"{track}_{split}_{seed}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    rng = np.random.default_rng(seed + TRACKS.index(track))
    data = [GENERATORS[track](rng) for _ in range(SIZES[split][track])]
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(data))
    return data


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(track: str, score_fn, data) -> dict:
    t0 = time.perf_counter()
    scores = [score_fn(sample) for sample, _ in data]
    seconds = time.perf_counter() - t0
    out = {"track": track, "n": len(data), "seconds_per_item": seconds / len(data)}
    if track == "cond":
        target = np.array([t for _, t in data])
        s = np.asarray(scores, float)
        out["spearman"] = float(stats.spearmanr(s, target, nan_policy="omit").statistic)
        out["auc_err_gt_0.05"] = auc((target > 0.05).astype(int), s)
        out["positive_fraction"] = float(np.mean(target > 0.05))
    elif track in POINT_LEVEL:
        y = np.concatenate([lab for _, lab in data])
        s = np.concatenate([np.asarray(sc, float).ravel() for sc in scores])
        out["auc"] = auc(y, s)
        precisions = []
        for sc, (_, lab) in zip(scores, data):
            k = int(lab.sum())
            top = np.argsort(-np.nan_to_num(np.asarray(sc, float), nan=np.inf))[:k]
            precisions.append(lab[top].mean())
        out["precision_at_k"] = float(np.mean(precisions))
    else:
        y = np.array([lab for _, lab in data])
        s = np.asarray(scores, float)
        out["auc"] = auc(y, s)
        out["tpr_at_5pct_fpr"] = tpr_at_fpr(y, s)
    return out


def load_score_fn(module_path: str):
    spec = importlib.util.spec_from_file_location(f"candidate_{abs(hash(module_path))}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(module_path).resolve().parent))
    spec.loader.exec_module(module)
    return module.score


# ---------------------------------------------------------------------------
# baselines (what the repository can do today)
# ---------------------------------------------------------------------------
def _tyler(P):
    from algorithms.utils import whiten_homogeneous_points
    return whiten_homogeneous_points(P)


def b_drift_single_tuple(s):
    t = np.arange(6)[None]
    return float(np.max(np.abs(log_abs_J_batch(s["A"], t) - log_abs_J_batch(s["B"], t)))) / s["delta"]


def b_drift_median_tuples(s):
    T = all_tuples(12)
    d = np.max(np.abs(log_abs_J_batch(s["A"], T) - log_abs_J_batch(s["B"], T)), axis=1)
    return float(np.median(d)) / s["delta"]


def fit_projective(A, B):
    """DLT for H with H a_i parallel to b_i (unit columns); returns H and per-point sine residuals."""
    rows = []
    for a, b in zip(A.T, B.T):
        Pb = np.eye(4) - np.outer(b, b) / (b @ b)
        rows.append(np.kron(Pb, a[None, :]))               # Pb @ H @ a = 0, H row-major
    _, _, vt = np.linalg.svd(np.vstack(rows))
    H = vt[-1].reshape(4, 4)
    HA = H @ A
    cos = np.abs(np.sum(HA * B, axis=0)) / (np.linalg.norm(HA, axis=0) * np.linalg.norm(B, axis=0))
    return H, np.sqrt(np.clip(1 - cos ** 2, 0, None))


def b_drift_registration(s):
    _, r = fit_projective(_tyler(s["A"]), _tyler(s["B"]))
    return float(np.sqrt(np.mean(r ** 2))) / s["delta"]


def b_cond_LS(s):
    from algorithms.conditioning.scores import compute_conditioning_score
    return float(np.log(compute_conditioning_score(s["P"], tuple(range(6)))))


def b_cond_kappa(s):
    from forensics.geometry.invariants import kappa_j
    return float(kappa_j(s["P"]))


def b_outlier_median_dJ(s):
    n = s["A"].shape[1]
    T = all_tuples(n)
    d = np.max(np.abs(log_abs_J_batch(s["A"], T) - log_abs_J_batch(s["B"], T)), axis=1)
    return np.array([np.median(d[np.any(T == i, axis=1)]) for i in range(n)])


def b_outlier_ransac_registration(s, iters: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    A, B = _tyler(s["A"]), _tyler(s["B"])
    n = A.shape[1]
    best, best_inliers = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 5, replace=False)
        H, _ = fit_projective(A[:, idx], B[:, idx])
        HA = H @ A
        cos = np.abs(np.sum(HA * B, axis=0)) / (np.linalg.norm(HA, axis=0) * np.linalg.norm(B, axis=0) + 1e-300)
        r = np.sqrt(np.clip(1 - cos ** 2, 0, None))
        inliers = int(np.sum(r < 0.02))
        if inliers > best_inliers:
            best, best_inliers = r < 0.02, inliers
    _, r = fit_projective(A[:, best], B[:, best]) if best.sum() >= 5 else (None, None)
    H, _ = fit_projective(A[:, best], B[:, best])
    HA = H @ A
    cos = np.abs(np.sum(HA * B, axis=0)) / (np.linalg.norm(HA, axis=0) * np.linalg.norm(B, axis=0) + 1e-300)
    return np.sqrt(np.clip(1 - cos ** 2, 0, None))


def robust_fundamental(x1, x2, seed=0, threshold=1e-3):
    """USAC-MAGSAC fundamental matrix; falls back to LMedS when OpenCV's USAC sampler asserts."""
    import cv2

    cv2.setRNGSeed(seed)
    try:
        F, mask = cv2.findFundamentalMat(x1, x2, cv2.USAC_MAGSAC, threshold, 0.9999)
    except cv2.error:
        F, mask = cv2.findFundamentalMat(x1, x2, cv2.FM_LMEDS)
    if F is None or F.shape[0] < 3:
        F, mask = cv2.findFundamentalMat(x1, x2, cv2.FM_8POINT)
    return F[:3], mask


def _multiview_reconstruct(views, seed=0):
    from forensics.video.multiview import bundle_adjust, cameras_from_fundamental, dlt_resection, dlt_triangulate

    F, _ = robust_fundamental(views[0], views[1], seed)
    P1, P2 = cameras_from_fundamental(F)
    X = dlt_triangulate([views[0], views[1]], [P1, P2])
    cams = [P1, P2] + [dlt_resection(X, views[v]) for v in range(2, len(views))]
    return bundle_adjust(cams, X, list(views))


def reconstruct_views(views, seed=0):
    """Projective reconstruction from >= 2 views (normalized coords): (cameras, X (4, N) unit columns, rms)."""
    return _multiview_reconstruct(list(views), seed)


def b_multiview_reprojection(s):
    from forensics.video.multiview import _project
    views = s["views"]
    cams, X, _ = _multiview_reconstruct(views)
    errors = np.stack([np.linalg.norm(_project(P, X) - views[v], axis=1) for v, P in enumerate(cams)])
    return errors.max(axis=0)


def b_multiview_subset_dJ(s, n_tuples: int = 800, seed: int = 0):
    views = s["views"]
    Xa = _tyler(_multiview_reconstruct([views[0], views[2], views[4]], seed)[1])
    Xb = _tyler(_multiview_reconstruct([views[1], views[3], views[5]], seed)[1])
    n = Xa.shape[1]
    rng = np.random.default_rng(seed)
    T = np.array([np.sort(rng.choice(n, 6, replace=False)) for _ in range(n_tuples)])
    d = np.max(np.abs(log_abs_J_batch(Xa, T) - log_abs_J_batch(Xb, T)), axis=1)
    return np.array([np.median(d[np.any(T == i, axis=1)]) if np.any(T == i) else 0.0 for i in range(n)])


BASELINES = {
    "drift": {"single_tuple_dJ": b_drift_single_tuple, "median_all_tuples_dJ": b_drift_median_tuples,
              "projective_registration_rms": b_drift_registration},
    "cond": {"log_L(S)": b_cond_LS, "kappa_J": b_cond_kappa},
    "outlier3d": {"median_dJ_per_point": b_outlier_median_dJ, "ransac_registration": b_outlier_ransac_registration},
    "multiview": {"reprojection_max": b_multiview_reprojection, "subset_dJ_median": b_multiview_subset_dJ},
    "nonrigid": {"median_dJ_per_point": b_outlier_median_dJ, "ransac_registration": b_outlier_ransac_registration},
}


def selftest():
    """Failure controls: a random scorer must give AUC ~ 0.5, an oracle 1.0, an anti-oracle 0."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    assert abs(auc(y, rng.random(2000)) - 0.5) < 0.05
    assert auc(y, y + 0.01 * rng.random(2000)) == 1.0
    assert auc(y, -y + 0.01 * rng.random(2000)) == 0.0
    # gauge must not change invariants
    P = unit_representatives(generate_generic_configuration(6, rng=rng))
    assert np.allclose(log_abs_J_batch(P, np.arange(6)[None]), log_abs_J_batch(gauge(P, rng), np.arange(6)[None]))
    print("selftest ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["baselines", "eval", "selftest", "build"])
    parser.add_argument("--track", choices=TRACKS)
    parser.add_argument("--module")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    tracks = [args.track] if args.track else list(TRACKS)
    for track in tracks:
        data = dataset(track, args.split, args.seed)
        if args.command == "build":
            print(track, len(data), "items")
        elif args.command == "baselines":
            for name, fn in BASELINES[track].items():
                print(json.dumps({"baseline": name, **evaluate(track, fn, data)}))
        else:
            print(json.dumps({"module": args.module, **evaluate(track, load_score_fn(args.module), data)}))


if __name__ == "__main__":
    main()
