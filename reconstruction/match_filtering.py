"""
reconstruction/match_filtering.py

Per-pair match filtering for the real-image benchmark (modes of colmap_capri_automator.py):
    "none"          matches unchanged
    "ransac"        USAC_MAGSAC fundamental-matrix inliers (control)
    "ransac_capri"  RANSAC inliers whose triangulated point lies in an accepted CAPRI tuple

Pairs are independent, so they are filtered in parallel worker processes (RANSAC and the
CAPRI layer run on the CPU). Each pair reseeds OpenCV's generator with `seed` before its
RANSAC call and passes `seed` to the CAPRI layer, so a pair's result does not depend on
the order of pairs or on the number of workers (tests/test_match_filtering.py).
"""

from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FILTER_MODES = ("none", "ransac", "ransac_capri")
RANSAC_ERROR = "ransac_error"   # flag returned by filter_pair when OpenCV raises


def default_workers() -> int:
    """All logical CPUs but one, so the machine stays responsive."""
    return max(1, (os.cpu_count() or 2) - 1)


def filter_pair(job: tuple) -> tuple[str, str, np.ndarray, int, bool | str]:
    """Filter one pair. Returns (name0, name1, kept index pairs, n after RANSAC, flag).

    flag: True if CAPRI accepted no tuple, RANSAC_ERROR if OpenCV raised, else False.
    """
    import cv2
    from pipeline.main_pipeline import run_capri_reliability_layer
    from reconstruction.triangulation import (
        estimate_projection_matrices_from_fundamental,
        triangulate_point_cloud,
    )

    (name0, name1, pairs, kp0, kp1, mode, seed, conditioning_threshold, max_tuples,
     empty_pair_policy, ransac_threshold_px, ransac_confidence) = job
    cv2.setNumThreads(1)  # one pair per process; avoid oversubscribing the CPU
    if mode == "none" or len(pairs) < 8:
        return name0, name1, pairs, len(pairs), False

    pts0 = kp0[pairs[:, 0]].astype(np.float64)
    pts1 = kp1[pairs[:, 1]].astype(np.float64)
    cv2.setRNGSeed(seed)
    try:
        F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.USAC_MAGSAC, ransac_threshold_px, ransac_confidence)
    except cv2.error:
        # USAC can fail an internal assertion on degenerate pairs instead of returning no
        # model; treated exactly like "no fundamental matrix" (counted in pairs_ransac_error)
        return name0, name1, pairs[:0], 0, RANSAC_ERROR
    if F is None or F.shape != (3, 3) or mask is None:
        return name0, name1, pairs[:0], 0, False
    inliers = mask.ravel().astype(bool)
    ransac_pairs = pairs[inliers]
    if mode == "ransac" or len(ransac_pairs) < 6:
        return name0, name1, ransac_pairs, len(ransac_pairs), False

    P1, P2 = estimate_projection_matrices_from_fundamental(F, pts0[inliers], pts1[inliers])
    points_3d = triangulate_point_cloud([pts0[inliers], pts1[inliers]], np.stack([P1, P2]))
    _, accepted = run_capri_reliability_layer(
        points_3d, np.ones(points_3d.shape[1]),
        conditioning_threshold=conditioning_threshold, max_tuples=max_tuples, seed=seed,
    )
    keep_rows = sorted({i for t in accepted for i in t})
    if not keep_rows:
        kept = ransac_pairs if empty_pair_policy == "keep_ransac" else ransac_pairs[:0]
        return name0, name1, kept, len(ransac_pairs), True
    return name0, name1, ransac_pairs[keep_rows], len(ransac_pairs), False


def _write_pair(out_group, name: str, matches0: np.ndarray, scores0: np.ndarray, keep: np.ndarray) -> int:
    """Write matches for one pair keeping only rows (idx0, idx1) in `keep`."""
    filtered0 = np.full_like(matches0, -1)
    filtered_scores = np.zeros_like(scores0)
    if len(keep):
        filtered0[keep[:, 0]] = keep[:, 1]
        filtered_scores[keep[:, 0]] = scores0[keep[:, 0]]
    group = out_group.create_group(name)
    group.create_dataset("matches0", data=filtered0)
    group.create_dataset("matching_scores0", data=filtered_scores)
    return int(len(keep))


def filter_matches(
    features_path: Path,
    matches_path: Path,
    output_path: Path,
    mode: str,
    seed: int = 42,
    conditioning_threshold: float = 1000.0,
    max_tuples: int = 150,
    empty_pair_policy: str = "keep_ransac",
    ransac_threshold_px: float = 3.0,
    ransac_confidence: float = 0.99,
    n_workers: int | None = None,
) -> dict:
    """Write a filtered copy of an hloc matches file and return statistics."""
    import h5py

    assert mode in FILTER_MODES, f"mode must be one of {FILTER_MODES}"
    assert empty_pair_policy in ("keep_ransac", "drop")
    n_workers = default_workers() if n_workers is None else n_workers
    assert n_workers >= 1
    stats = {"mode": mode, "seed": seed, "conditioning_threshold": conditioning_threshold,
             "empty_pair_policy": empty_pair_policy, "n_workers": n_workers, "pairs": 0,
             "matches_in": 0, "matches_after_ransac": 0, "matches_out": 0, "pairs_capri_empty": 0,
             "pairs_ransac_error": 0}

    jobs, raw = [], {}
    with h5py.File(features_path, "r") as f_feat, h5py.File(matches_path, "r") as f_match:
        keypoints = {name: f_feat[name]["keypoints"][:] for name in f_feat.keys()}
        for name0 in f_match.keys():
            for name1 in f_match[name0].keys():
                matches0 = f_match[name0][name1]["matches0"][:]
                scores0 = f_match[name0][name1]["matching_scores0"][:]
                valid = matches0 > -1
                pairs = np.stack([np.where(valid)[0], matches0[valid]], axis=-1)
                raw[(name0, name1)] = (matches0, scores0)
                stats["pairs"] += 1
                stats["matches_in"] += len(pairs)
                jobs.append((name0, name1, pairs, keypoints[name0], keypoints[name1], mode, seed,
                             conditioning_threshold, max_tuples, empty_pair_policy,
                             ransac_threshold_px, ransac_confidence))

    if n_workers == 1 or mode == "none":
        results = [filter_pair(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(filter_pair, jobs, chunksize=max(1, len(jobs) // (8 * n_workers))))

    with h5py.File(output_path, "w") as f_out:
        for name0, name1, kept, n_after_ransac, flag in results:
            matches0, scores0 = raw[(name0, name1)]
            group = f_out[name0] if name0 in f_out else f_out.create_group(name0)
            stats["matches_after_ransac"] += n_after_ransac
            stats["pairs_ransac_error"] += int(flag == RANSAC_ERROR)
            stats["pairs_capri_empty"] += int(flag is True)
            stats["matches_out"] += _write_pair(group, name1, matches0, scores0, kept)
    return stats
