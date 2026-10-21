"""
forensics/structure/lines.py

Line-segment extraction and the segment noise model.

Noise model: a segment of length L pixels fitted to n = L / l_c effectively
independent edge pixels, each localized with standard deviation sigma_p, has
orientation variance (orthogonal regression on equally spaced points)
    sigma_theta^2 = 12 sigma_p^2 / (n L^2) = 12 sigma_p^2 l_c / L^3,
and perpendicular midpoint variance sigma_p^2 l_c / L. Long segments are far
more informative (sigma_theta ~ L^{-3/2}).

Coordinates: pixels are mapped to normalized coordinates by centring at the
image centre and dividing by max(H, W) / 2 (Hartley-style conditioning).
Images are never resized before analysis.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class Segments:
    """Segments in normalized coordinates with per-segment noise."""
    p1: np.ndarray          # (n, 2) endpoint 1
    p2: np.ndarray          # (n, 2) endpoint 2
    length_px: np.ndarray   # (n,)
    sigma_theta: np.ndarray  # (n,) orientation std (radians)
    sigma_perp: np.ndarray   # (n,) perpendicular midpoint std (normalized units)
    scale: float            # pixels per normalized unit
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.length_px)

    @property
    def midpoints(self) -> np.ndarray:
        return 0.5 * (self.p1 + self.p2)

    @property
    def theta(self) -> np.ndarray:
        """Orientation in [0, pi)."""
        d = self.p2 - self.p1
        return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)

    @property
    def homogeneous_lines(self) -> np.ndarray:
        """(n, 3) unit-norm lines l = p1 x p2."""
        ones = np.ones((len(self), 1))
        lines = np.cross(np.hstack([self.p1, ones]), np.hstack([self.p2, ones]))
        return lines / np.linalg.norm(lines, axis=1, keepdims=True)

    def subset(self, index: np.ndarray) -> "Segments":
        return Segments(self.p1[index], self.p2[index], self.length_px[index], self.sigma_theta[index],
                        self.sigma_perp[index], self.scale, self.width, self.height)


def orientation_sigma(length_px: np.ndarray, edge_sigma_px: float, corr_length_px: float) -> np.ndarray:
    """sigma_theta (radians) = sqrt(12 sigma_p^2 l_c / L^3)."""
    length_px = np.asarray(length_px, dtype=float)
    return np.sqrt(12.0 * edge_sigma_px ** 2 * corr_length_px / length_px ** 3)


def perpendicular_sigma_px(length_px: np.ndarray, edge_sigma_px: float, corr_length_px: float) -> np.ndarray:
    """Perpendicular std of the fitted midpoint, sqrt(sigma_p^2 l_c / L), in pixels."""
    return np.sqrt(edge_sigma_px ** 2 * corr_length_px / np.asarray(length_px, dtype=float))


def normalization(width: int, height: int) -> tuple[np.ndarray, float]:
    """(centre in pixels, pixels per normalized unit)."""
    return np.array([width / 2.0, height / 2.0]), max(width, height) / 2.0


def segments_from_endpoints(endpoints_px: np.ndarray, width: int, height: int, config: dict) -> Segments:
    """Build Segments from (n, 4) pixel endpoints, applying the length and precision filters."""
    cfg = config["lines"]
    endpoints_px = np.asarray(endpoints_px, dtype=float).reshape(-1, 4)
    length = np.linalg.norm(endpoints_px[:, 2:] - endpoints_px[:, :2], axis=1)
    keep = length >= cfg["min_len_frac"] * max(width, height)
    endpoints_px, length = endpoints_px[keep], length[keep]
    order = np.argsort(-length)[: cfg["max_segments"]]
    endpoints_px, length = endpoints_px[order], length[order]
    sigma_theta = orientation_sigma(length, cfg["edge_sigma_px"], cfg["corr_length_px"])
    precise = sigma_theta <= np.deg2rad(cfg["max_sigma_theta_deg"])
    endpoints_px, length, sigma_theta = endpoints_px[precise], length[precise], sigma_theta[precise]
    centre, scale = normalization(width, height)
    return Segments(
        p1=(endpoints_px[:, :2] - centre) / scale,
        p2=(endpoints_px[:, 2:] - centre) / scale,
        length_px=length,
        sigma_theta=sigma_theta,
        sigma_perp=perpendicular_sigma_px(length, cfg["edge_sigma_px"], cfg["corr_length_px"]) / scale,
        scale=scale, width=width, height=height,
    )


def detect_lsd(gray: np.ndarray, config: dict) -> np.ndarray:
    """LSD line segments (n, 4) in pixels at native resolution."""
    import cv2

    cfg = config["lines"]
    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD, cfg["lsd_scale"], cfg["lsd_sigma_scale"], cfg["lsd_quant"],
        cfg["lsd_angle_tolerance_deg"], 0.0, cfg["lsd_density"], 1024)
    lines = detector.detect(gray)[0]
    return np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)


def extract_segments(gray: np.ndarray, config: dict) -> Segments:
    height, width = gray.shape[:2]
    return segments_from_endpoints(detect_lsd(gray, config), width, height, config)
