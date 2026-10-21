"""
synthetic_data/injection.py

Multi-view scenes with point-level inconsistency injected at a controlled dose
(docs/prereg/evaluation_plan.md, injection grid = injection harness of the evaluation shortlist).

A scene is N points seen by V calibrated cameras on an arc (focal 800 px,
1280 x 720 px, as forensics/video/synthetic.py). View v sees world points
W_v (N, 3); for a rigid scene W_v = W for every v. A family changes W_v:

  rigid               W_v = W.
  global_projective   views of subset B see H(W) for one 3D projective map H
                      close to the identity (the provable blind spot: the
                      images stay consistent with a rigid scene under other
                      cameras, so no test on tracks alone can see it).
  jitter              a fraction of the points gets an independent random 3D
                      offset in every view.
  drift               a fraction of the points moves along a fixed direction,
                      linearly in time (v - (V - 1) / 2).
  warp                points near a random centre move by a smooth (Gaussian)
                      field whose amplitude varies over time as sin(pi v / (V - 1)).
  swap                a fraction of the tracks swap identity with their nearest
                      neighbour in views V/2 .. V-1 (dose = fraction swapped).
  planar              a rigid scene on a plane (degenerate; must abstain).

The dose of global_projective, jitter, drift and warp is the induced pixel displacement: the RMS over the moved
tracks and all views of |project(W_v) - project(W)| in pixels, before noise.
The world offset is scaled so the induced displacement equals the dose; the
scale is found by bisection on the exact projection, so it holds to 1e-3 px.
"""

from __future__ import annotations

import numpy as np

WIDTH, HEIGHT, FOCAL = 1280, 720, 800.0
FAMILIES = ("rigid", "global_projective", "jitter", "drift", "warp", "swap", "planar")
MOVED_FRACTION = 0.3


def look_at(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    return rotation, -rotation @ position


def arc_cameras(n_views: int, arc_deg: float = 30.0, radius: float = 6.0, start_deg: float | None = None) -> list[np.ndarray]:
    """3 x 4 pixel cameras K [R | t] on an arc around the origin."""
    K = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1.0]])
    first = -arc_deg / 2 if start_deg is None else start_deg
    cameras = []
    for angle in np.deg2rad(first + np.linspace(0.0, arc_deg, n_views)):
        rotation, translation = look_at(np.array([radius * np.sin(angle), -0.5, -radius * np.cos(angle)]), np.zeros(3))
        cameras.append(K @ np.hstack([rotation, translation[:, None]]))
    return cameras


def project(camera: np.ndarray, world: np.ndarray) -> np.ndarray:
    """(N, 3) world -> (N, 2) pixels."""
    homogeneous = np.hstack([world, np.ones((len(world), 1))]) @ camera.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def scene_points(rng: np.random.Generator, n_points: int, planar: bool = False) -> np.ndarray:
    points = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    if planar:
        points[:, 2] = 0.3 * points[:, 0] - 0.2 * points[:, 1]
    return points


def _displacement(family: str, world: np.ndarray, n_views: int, rng: np.random.Generator):
    """(function scale -> per-view world points (V, N, 3), moved mask (N,)) for global_projective/jitter/drift/warp."""
    n = len(world)
    moved = np.zeros(n, dtype=bool)
    if family == "global_projective":
        moved[:] = True
        perturbation = rng.standard_normal((4, 4))
        homogeneous = np.hstack([world, np.ones((n, 1))])

        def projective(scale: float) -> np.ndarray:
            mapped = homogeneous @ (np.eye(4) + scale * perturbation).T
            per_view = np.repeat(world[None], n_views, axis=0)
            per_view[1::2] = mapped[:, :3] / mapped[:, 3:4]       # exact H(W) in the views of subset B
            return per_view
        return projective, moved
    offsets = np.zeros((n_views, n, 3))
    count = max(1, int(round(MOVED_FRACTION * n)))
    if family == "warp":
        centre = world[rng.integers(n)]
        distance = np.linalg.norm(world - centre, axis=1)
        moved[np.argsort(distance)[:count]] = True
        weight = np.exp(-0.5 * (distance / np.sort(distance)[count - 1]) ** 2) * moved
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        for v in range(n_views):
            offsets[v] = np.sin(np.pi * v / (n_views - 1)) * weight[:, None] * direction[None, :]
    else:
        moved[rng.choice(n, count, replace=False)] = True
        if family == "jitter":
            offsets[:, moved] = rng.standard_normal((n_views, count, 3)) / np.sqrt(3.0)
        elif family == "drift":
            direction = rng.standard_normal((count, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            for v in range(n_views):
                offsets[v, moved] = (v - (n_views - 1) / 2) / ((n_views - 1) / 2) * direction
        else:
            raise ValueError(family)
    return (lambda scale: world[None] + scale * offsets), moved


def _induced_px(cameras, world, per_view, moved) -> float:
    if not moved.any():
        return 0.0
    shifts = [project(P, per_view[v][moved]) - project(P, world[moved]) for v, P in enumerate(cameras)]
    return float(np.sqrt(np.mean(np.sum(np.concatenate(shifts) ** 2, axis=1))))


def _scale_for_dose(cameras, world, displace, moved, dose_px: float) -> float:
    """Bisection on the scale so that the induced RMS displacement equals dose_px."""
    def induced(scale):
        return _induced_px(cameras, world, displace(scale), moved)

    low, high = 0.0, 1e-4
    while induced(high) < dose_px:
        high *= 2.0
        assert high < 1e3, "dose not reachable"
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if induced(middle) < dose_px else (low, middle)
    return 0.5 * (low + high)


def make_window(rng: np.random.Generator, family: str, dose: float, noise_px: float, n_points: int = 40,
                n_views: int = 6, cameras: list[np.ndarray] | None = None, world: np.ndarray | None = None) -> dict:
    """
    One window of V views. Returns {"tracks": (V, N, 2) noisy pixels, "clean": noise-free pixels,
    "moved": (N,) bool, "induced_px": realised dose, "family", "dose"}. Draw order: points (unless
    given), family offsets, then the pixel noise.
    """
    assert family in FAMILIES and dose >= 0 and noise_px >= 0
    cameras = arc_cameras(n_views) if cameras is None else cameras
    if world is None:
        world = scene_points(rng, n_points, planar=(family == "planar"))
    n = len(world)
    per_view = np.repeat(world[None], n_views, axis=0)
    moved = np.zeros(n, dtype=bool)
    if family in ("global_projective", "jitter", "drift", "warp"):
        displace, moved = _displacement(family, world, n_views, rng)
        if dose > 0:
            per_view = displace(_scale_for_dose(cameras, world, displace, moved, dose))
    clean = np.stack([project(P, per_view[v]) for v, P in enumerate(cameras)])
    if family == "swap" and dose > 0:
        count = max(1, int(round(dose * n)))
        image = clean[0]
        order = rng.permutation(n)
        used = np.zeros(n, dtype=bool)
        pairs = []
        for i in order:
            if len(pairs) * 2 >= count or used[i]:
                continue
            distance = np.linalg.norm(image - image[i], axis=1)
            distance[used | (np.arange(n) == i)] = np.inf
            j = int(np.argmin(distance))
            if np.isfinite(distance[j]):
                pairs.append((i, j))
                used[[i, j]] = True
        for i, j in pairs:
            moved[[i, j]] = True
            clean[n_views // 2:, [i, j]] = clean[n_views // 2:, [j, i]]
    induced = (float(np.sqrt(np.mean(np.sum((clean[:, moved] - np.stack([project(P, world[moved]) for P in cameras])) ** 2,
                                            axis=2)))) if moved.any() else 0.0)
    tracks = clean + rng.normal(0.0, noise_px, clean.shape)
    return {"tracks": tracks, "clean": clean, "moved": moved, "induced_px": induced, "family": family, "dose": dose}
