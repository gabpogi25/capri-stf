"""
forensics/structure/synthetic.py

Synthetic line segments for validating the image-branch statistics (stage S1,
tests T08/T09). Segments are drawn so that, before noise, each family passes
exactly through its vanishing point; noise follows the segment model of
lines.py (orientation std sigma_theta(L), perpendicular midpoint std).
This is a check of the statistics, NOT evidence that real images follow the
noise model (that is the real-media null gate).
"""

from __future__ import annotations
import numpy as np

from forensics.structure.lines import Segments, orientation_sigma, perpendicular_sigma_px


def synthetic_segments(
    rng: np.random.Generator,
    vanishing_points_px: list[tuple[float, float]] | None = None,
    segments_per_family: int = 40,
    length_range_px: tuple[float, float] = (120.0, 400.0),
    width: int = 1600,
    height: int = 1200,
    edge_sigma_px: float = 0.7,
    corr_length_px: float = 2.0,
    inconsistency_deg: float = 0.0,
) -> Segments:
    """
    Segments consistent with the given vanishing points (pixels). With
    inconsistency_deg > 0 every segment is additionally rotated by a random
    angle of that standard deviation (a geometrically inconsistent image).
    """
    if vanishing_points_px is None:
        vanishing_points_px = [(width * 2.5, height * 0.45), (-width * 1.6, height * 0.55), (width * 0.5, -height * 4.0)]
    scale = max(width, height) / 2.0
    centre = np.array([width / 2.0, height / 2.0])
    p1s, p2s, lengths = [], [], []
    for vp in vanishing_points_px:
        vp = np.asarray(vp, dtype=float)
        for _ in range(segments_per_family):
            midpoint = rng.uniform([0.1 * width, 0.1 * height], [0.9 * width, 0.9 * height])
            length = rng.uniform(*length_range_px)
            direction = vp - midpoint
            angle = np.arctan2(direction[1], direction[0])
            angle += rng.normal(0.0, orientation_sigma(np.array([length]), edge_sigma_px, corr_length_px)[0])
            if inconsistency_deg > 0:
                angle += rng.normal(0.0, np.deg2rad(inconsistency_deg))
            normal = np.array([-np.sin(angle), np.cos(angle)])
            midpoint = midpoint + normal * rng.normal(0.0, perpendicular_sigma_px(np.array([length]), edge_sigma_px,
                                                                                     corr_length_px)[0])
            half = 0.5 * length * np.array([np.cos(angle), np.sin(angle)])
            p1s.append(midpoint - half)
            p2s.append(midpoint + half)
            lengths.append(length)
    lengths = np.asarray(lengths)
    return Segments(
        p1=(np.asarray(p1s) - centre) / scale, p2=(np.asarray(p2s) - centre) / scale,
        length_px=lengths, sigma_theta=orientation_sigma(lengths, edge_sigma_px, corr_length_px),
        sigma_perp=perpendicular_sigma_px(lengths, edge_sigma_px, corr_length_px) / scale,
        scale=scale, width=width, height=height)


def render_perspective_scene(rng: np.random.Generator, width: int = 1280, height: int = 960,
                             distortion_k1: float = 0.0, noise_level: float = 2.0) -> np.ndarray:
    """
    Grayscale rendering of a box room (floor/wall grid lines) through a pinhole
    camera, optionally with radial distortion x_d = x (1 + k1 r^2) applied to
    the rendered image. A functional fixture for the image pipeline only.
    """
    import cv2

    focal = 0.9 * width
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    yaw, pitch = rng.uniform(-0.35, 0.35), rng.uniform(-0.15, 0.05)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R = Rx @ Ry
    t = np.array([0.0, 0.0, 0.0])
    image = np.full((height, width), 128.0)
    shade = {"floor": 90.0, "left": 150.0, "right": 175.0, "back": 120.0}

    def project(points):
        cam = points @ R.T + t
        valid = cam[:, 2] > 0.2
        uv = (K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3], valid

    lines = []
    span = np.linspace(-3, 3, 7)
    depth = np.linspace(2.5, 14, 9)
    for x in span:                              # floor lines along depth (vanishing in z)
        lines.append((np.array([x, 1.5, 2.5]), np.array([x, 1.5, 14.0])))
    for z in depth:                             # floor lines across
        lines.append((np.array([-3, 1.5, z]), np.array([3, 1.5, z])))
    for y in np.linspace(-1.5, 1.5, 5):         # side-wall horizontal lines
        lines.append((np.array([-3, y, 2.5]), np.array([-3, y, 14.0])))
        lines.append((np.array([3, y, 2.5]), np.array([3, y, 14.0])))
    for z in depth[::2]:                        # side-wall verticals
        lines.append((np.array([-3, -1.5, z]), np.array([-3, 1.5, z])))
        lines.append((np.array([3, -1.5, z]), np.array([3, 1.5, z])))
    for x in span:                              # back-wall verticals
        lines.append((np.array([x, -1.5, 14.0]), np.array([x, 1.5, 14.0])))
    for key, value in shade.items():
        pass
    canvas = image.copy()
    for a, b in lines:
        pts = np.linspace(a, b, 200)
        uv, valid = project(pts)
        uv = uv[valid]
        if len(uv) < 2:
            continue
        cv2.polylines(canvas, [np.round(uv * 16).astype(np.int32).reshape(-1, 1, 2)], False, 30.0, 2,
                      lineType=cv2.LINE_AA, shift=4)
    if distortion_k1 != 0.0:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        xn, yn = (xs - width / 2) / focal, (ys - height / 2) / focal
        r2 = xn ** 2 + yn ** 2
        map_x = (xn * (1 + distortion_k1 * r2)) * focal + width / 2
        map_y = (yn * (1 + distortion_k1 * r2)) * focal + height / 2
        canvas = cv2.remap(canvas.astype(np.float32), map_x.astype(np.float32), map_y.astype(np.float32),
                           cv2.INTER_LINEAR, borderValue=128.0)
    canvas = canvas + rng.normal(0.0, noise_level, canvas.shape)
    return np.clip(canvas, 0, 255).astype(np.uint8)
