"""
forensics/structure/collinearity.py

Bend test (constraint: points of one straight 3D edge image to collinear
points under a pinhole camera).

For a long segment, sub-pixel edge points are sampled along it: at every
pixel step the gradient magnitude is read along the segment normal and the
peak is located by a parabolic fit. The support is split into two disjoint
halves, a line is fitted to each by orthogonal regression, and
    z = (theta_1 - theta_2) / sqrt(sigma_theta_1^2 + sigma_theta_2^2)
with sigma_theta from the segment noise model of lines.py. Under a straight
edge and the noise model, z ~ N(0, 1).

Lens distortion bends straight edges, so this test also responds to
distortion. That is a confound to measure (authentic fisheye / wide-angle
media in the stress set), not to hide.
"""

from __future__ import annotations
import numpy as np

from forensics.structure.lines import orientation_sigma


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    import cv2

    smoothed = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), 1.0)
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    return (image[y0, x0] * (1 - dx) * (1 - dy) + image[y0, x0 + 1] * dx * (1 - dy)
            + image[y0 + 1, x0] * (1 - dx) * dy + image[y0 + 1, x0 + 1] * dx * dy)


def edge_support_points(gradient: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                        search_px: float = 3.0, margin_frac: float = 0.05) -> np.ndarray:
    """Sub-pixel edge points (m, 2) along the segment p1 -> p2 (pixel coordinates)."""
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length < 2:
        return np.zeros((0, 2))
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    ts = np.arange(margin_frac * length, (1 - margin_frac) * length, 1.0)
    offsets = np.arange(-search_px, search_px + 1e-9, 1.0)
    points = []
    for t in ts:
        base = p1 + t * direction
        xs = base[0] + offsets * normal[0]
        ys = base[1] + offsets * normal[1]
        profile = _bilinear(gradient, xs, ys)
        k = int(np.argmax(profile))
        if k == 0 or k == len(offsets) - 1:
            continue
        left, centre, right = profile[k - 1], profile[k], profile[k + 1]
        denominator = left - 2 * centre + right
        shift = 0.5 * (left - right) / denominator if denominator < 0 else 0.0
        points.append(base + (offsets[k] + shift) * normal)
    return np.asarray(points)


def orthogonal_regression_angle(points: np.ndarray) -> float:
    """Direction angle (mod pi) of the total-least-squares line through the points."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    d = vt[0]
    return float(np.mod(np.arctan2(d[1], d[0]), np.pi))


def bend_test(gray: np.ndarray, endpoints_px: np.ndarray, config: dict) -> list[dict]:
    """z-scores of the two-half orientation difference for each sufficiently long segment."""
    cfg_lines, cfg = config["lines"], config["collinearity"]
    height, width = gray.shape[:2]
    gradient = _gradient_magnitude(gray)
    min_length = cfg["min_len_frac_bend"] * max(width, height)
    results = []
    for x1, y1, x2, y2 in np.asarray(endpoints_px, dtype=float).reshape(-1, 4):
        p1, p2 = np.array([x1, y1]), np.array([x2, y2])
        if np.linalg.norm(p2 - p1) < min_length:
            continue
        support = edge_support_points(gradient, p1, p2)
        if len(support) < cfg["min_support_px"]:
            continue
        half = len(support) // 2
        first, second = support[:half], support[half:]
        lengths = [float(np.linalg.norm(part[-1] - part[0])) for part in (first, second)]
        sigmas = orientation_sigma(np.array(lengths), cfg_lines["edge_sigma_px"], cfg_lines["corr_length_px"])
        if np.any(sigmas > np.deg2rad(cfg["max_sigma_theta_deg"])):
            continue
        difference = orthogonal_regression_angle(first) - orthogonal_regression_angle(second)
        difference = (difference + np.pi / 2) % np.pi - np.pi / 2
        variance = float(np.sum(sigmas ** 2))
        results.append({"kind": "bend", "residual": float(difference), "variance": variance,
                        "z": float(difference / np.sqrt(variance)), "length_px": float(sum(lengths))})
    return results
