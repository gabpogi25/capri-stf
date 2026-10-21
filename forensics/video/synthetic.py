"""
forensics/video/synthetic.py

Synthetic rigid-scene generator for the multi-view branch (stage S6a):
N points uniform in a cube, six cameras on an arc looking at the cube,
focal 800 px, 1280 x 720 px, Gaussian pixel noise. Optional non-rigid
injection moves chosen points in the views of subset B only.

Used for three checks before any real data: (i) noise-free Delta_J ~ 0,
(ii) the noisy null spread, (iii) |Delta_J| increasing with injected
displacement. These validate the code, not the claim that generated videos
violate rigidity.
"""

from __future__ import annotations
import numpy as np

WIDTH, HEIGHT, FOCAL = 1280, 720, 800.0


def look_at(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    return R, -R @ position


def synthetic_rigid_views(rng: np.random.Generator, n_points: int = 60, n_views: int = 6,
                          noise_px: float = 0.5, arc_deg: float = 30.0, displaced: dict | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (tracks in pixels, shape (n_views, N, 2); true 3D points (N, 3)).
    displaced = {"indices": [...], "views": [...], "offset": float}: those points
    are moved by `offset` (scene units, random direction) in the listed views only.
    """
    points = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])
    angles = np.deg2rad(np.linspace(-arc_deg / 2, arc_deg / 2, n_views))
    radius = 6.0
    tracks = np.zeros((n_views, n_points, 2))
    direction = rng.standard_normal((n_points, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    for v, angle in enumerate(angles):
        position = np.array([radius * np.sin(angle), -0.5, -radius * np.cos(angle)])
        R, t = look_at(position, np.zeros(3))
        world = points.copy()
        if displaced and v in displaced["views"]:
            idx = np.asarray(displaced["indices"])
            world[idx] += displaced["offset"] * direction[idx]
        cam = world @ R.T + t
        uv = (K @ cam.T).T
        tracks[v] = uv[:, :2] / uv[:, 2:3] + rng.normal(0.0, noise_px, (n_points, 2))
    return tracks, points


def normalize_pixels(tracks_px: np.ndarray, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    centre = np.array([width / 2.0, height / 2.0])
    return (tracks_px - centre) / (max(width, height) / 2.0)


def _texture(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    import cv2

    texture = np.zeros((size, size))
    for sigma, weight in ((1.5, 0.5), (4.0, 0.3), (12.0, 0.2)):
        texture += weight * cv2.GaussianBlur(rng.uniform(0, 255, (size, size)), (0, 0), sigma)
    texture = (texture - texture.min()) / (np.ptp(texture) + 1e-9) * 255.0
    return texture.astype(np.float32)


def render_synthetic_video(path: str, rng: np.random.Generator, n_frames: int = 90, fps: float = 30.0,
                           width: int = 960, height: int = 540, warp_amplitude: float = 0.0) -> None:
    """
    Write an mp4 of a textured box room seen by a camera translating sideways
    with a slow yaw. With warp_amplitude > 0 every frame is additionally warped
    by a radial distortion whose coefficient oscillates over time
    (k1(t) = warp_amplitude * sin(2 pi t / n_frames)), so the frames are no
    longer views of one rigid scene through one camera. Functional fixture
    for the video pipeline only.
    """
    import cv2

    focal = 0.8 * width
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    planes = [  # (origin, axis_u, axis_v, extent_u, extent_v)
        (np.array([-4.0, -2.0, 12.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), 8.0, 4.0),   # back wall
        (np.array([-4.0, -2.0, 1.0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), 11.0, 4.0),   # left wall
        (np.array([4.0, -2.0, 1.0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), 11.0, 4.0),    # right wall
        (np.array([-4.0, 2.0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), 8.0, 11.0),    # floor
    ]
    textures = [_texture(rng) for _ in planes]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    for frame_index in range(n_frames):
        s = frame_index / max(n_frames - 1, 1)
        yaw = np.deg2rad(-6.0 + 12.0 * s)
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        centre = np.array([-2.5 + 5.0 * s, 0.0, -1.0])
        t = -R @ centre
        canvas = np.full((height, width), 128.0, dtype=np.float32)
        for (origin, axis_u, axis_v, ext_u, ext_v), texture in zip(planes, textures):
            size = texture.shape[0]
            H = K @ np.column_stack([R @ axis_u * ext_u / size, R @ axis_v * ext_v / size, R @ origin + t])
            warped = cv2.warpPerspective(texture, H, (width, height), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.ones_like(texture), H, (width, height), flags=cv2.INTER_NEAREST)
            canvas = np.where(mask > 0.5, warped, canvas)
        if warp_amplitude > 0:
            k1 = warp_amplitude * np.sin(2 * np.pi * frame_index / n_frames)
            xn, yn = (xs - width / 2) / focal, (ys - height / 2) / focal
            factor = 1 + k1 * (xn ** 2 + yn ** 2)
            map_x = (xn * factor * focal + width / 2).astype(np.float32)
            map_y = (yn * factor * focal + height / 2).astype(np.float32)
            canvas = cv2.remap(canvas, map_x, map_y, cv2.INTER_LINEAR, borderValue=128.0)
        frame = np.clip(canvas + rng.normal(0, 1.5, canvas.shape), 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
