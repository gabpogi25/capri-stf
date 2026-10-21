"""
forensics/video/tracking.py

Point tracking through a window of views and rigid segmentation.

Tracker: pyramidal Lucas-Kanade on a grid x grid lattice of query points in
the first view, tracked view to view. Confidence is the forward-backward
consistency exp(-e_fb), where e_fb is the distance between a point and its
position after tracking forward one view and back again. A track is kept only
if its confidence is at least track_conf_min in every step and it stays
inside the frame. (KLT is the dependency-free default; a learned point
tracker, or any tracker producing (N, V, 2) tracks with confidences, can
replace it.)

Rigid segmentation: for every pair of views a USAC-MAGSAC fundamental matrix
is fitted; tracks that are inliers in at least rigid_inlier_frac of the pairs
are kept as the static (rigid) background.
"""

from __future__ import annotations
from itertools import combinations

import numpy as np


def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def track_window(frames: list[np.ndarray], grid: int, conf_min: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (tracks (N, V, 2) in pixels, confidence (N, V)) for tracks that survive the whole window."""
    import cv2

    grays = [_to_gray(f) for f in frames]
    h, w = grays[0].shape
    ys, xs = np.meshgrid(np.linspace(0.05 * h, 0.95 * h, grid), np.linspace(0.05 * w, 0.95 * w, grid), indexing="ij")
    points = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tracks = [points.copy()]
    confidence = [np.ones(len(points))]
    alive = np.ones(len(points), dtype=bool)
    current = points
    for a, b in zip(grays[:-1], grays[1:]):
        forward, status_f, _ = cv2.calcOpticalFlowPyrLK(a, b, current.reshape(-1, 1, 2), None, **lk)
        backward, status_b, _ = cv2.calcOpticalFlowPyrLK(b, a, forward, None, **lk)
        forward, backward = forward.reshape(-1, 2), backward.reshape(-1, 2)
        fb_error = np.linalg.norm(backward - current, axis=1)
        conf = np.exp(-fb_error)
        inside = (forward[:, 0] >= 0) & (forward[:, 0] < w) & (forward[:, 1] >= 0) & (forward[:, 1] < h)
        alive &= (status_f.ravel() == 1) & (status_b.ravel() == 1) & (conf >= conf_min) & inside
        tracks.append(forward)
        confidence.append(conf)
        current = forward
    tracks = np.stack(tracks, axis=1)[alive]
    confidence = np.stack(confidence, axis=1)[alive]
    return tracks.astype(float), confidence


def _usac(estimator, *args):
    """Call an OpenCV USAC estimator; (None, None) when it fails an internal assertion.

    USAC_MAGSAC can raise cv2.error (!model.empty()) on degenerate input instead of returning
    no model; that case is treated exactly like "no model" (no inliers).
    """
    import cv2

    try:
        return estimator(*args)
    except cv2.error:
        return None, None


def rigid_inliers(tracks: np.ndarray, threshold_px: float, confidence: float, min_fraction: float,
                  seed: int = 0) -> np.ndarray:
    """Boolean mask of tracks that are fundamental-matrix inliers in >= min_fraction of view pairs."""
    import cv2

    n_tracks, n_views = tracks.shape[:2]
    if n_tracks < 8:
        return np.zeros(n_tracks, dtype=bool)
    cv2.setRNGSeed(seed)
    counts = np.zeros(n_tracks)
    pairs = list(combinations(range(n_views), 2))
    for a, b in pairs:
        F, mask = _usac(cv2.findFundamentalMat, tracks[:, a], tracks[:, b], cv2.USAC_MAGSAC, threshold_px, confidence)
        if F is not None and mask is not None:
            counts += mask.ravel().astype(bool)
    return counts >= min_fraction * len(pairs)


def parallax_ok(tracks: np.ndarray, width: int, height: int, min_disp_frac: float, min_f_over_h: float,
                threshold_px: float, seed: int = 0) -> tuple[bool, dict]:
    """
    Parallax gate (Section 8.5.5) on the first-to-last view pair: median track
    displacement >= min_disp_frac of the diagonal and inliers(F) / inliers(H)
    >= min_f_over_h (rejects pure rotation, zoom and planar-dominant scenes).
    """
    import cv2

    first, last = tracks[:, 0], tracks[:, -1]
    displacement = float(np.median(np.linalg.norm(last - first, axis=1)))
    diagonal = float(np.hypot(width, height))
    cv2.setRNGSeed(seed)
    _, mask_f = _usac(cv2.findFundamentalMat, first, last, cv2.USAC_MAGSAC, threshold_px, 0.999)
    _, mask_h = _usac(cv2.findHomography, first, last, cv2.USAC_MAGSAC, threshold_px)
    inliers_f = int(mask_f.sum()) if mask_f is not None else 0
    inliers_h = int(mask_h.sum()) if mask_h is not None else 0
    ratio = inliers_f / max(inliers_h, 1)
    info = {"median_displacement_frac": displacement / diagonal, "inliers_f": inliers_f,
            "inliers_h": inliers_h, "f_over_h": ratio}
    return displacement >= min_disp_frac * diagonal and ratio >= min_f_over_h, info
