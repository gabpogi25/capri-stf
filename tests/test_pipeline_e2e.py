"""
End-to-end checks of the 3D-reconstruction pipeline (pipeline/main_pipeline.py)
on rendered multi-view images, and of the frame whitening it relies on.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.utils import whiten_homogeneous_points


def test_whitening_preserves_invariants_and_fixes_conditioning_chart():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.uniform(-1, 1, (3, 12)), np.ones((1, 12))])
    skew = np.diag([1.0, 1.0, 1.0, 1e-4]) @ np.linalg.qr(rng.standard_normal((4, 4)))[0]
    skewed = skew @ points
    t = (0, 1, 2, 3, 4, 5)
    a, b = evaluate_six_point_invariants(points, t), evaluate_six_point_invariants(skewed, t)
    assert a is not None and b is not None
    assert np.allclose(a, b, rtol=1e-6)
    # L(S) is chart-dependent: the skewed frame inflates it; whitening both frames gives the same value
    assert compute_conditioning_score(skewed, t) > 10 * compute_conditioning_score(points, t)
    assert compute_conditioning_score(whiten_homogeneous_points(skewed), t) == pytest.approx(
        compute_conditioning_score(whiten_homogeneous_points(points), t), rel=1e-6)


def test_pipeline_end_to_end_on_rendered_views(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from forensics.video.synthetic import render_synthetic_video
    from pipeline.main_pipeline import run_pipeline

    video = tmp_path / "clip.mp4"
    render_synthetic_video(str(video), np.random.default_rng(3), n_frames=40)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    capture, index, written = cv2.VideoCapture(str(video)), 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % 8 == 0:
            cv2.imwrite(str(image_dir / f"view_{written:02d}.png"), frame)
            written += 1
        index += 1
    assert written >= 4

    out = tmp_path / "out"
    run_pipeline(image_dir, out, seed=1)
    for name in ("reconstruction.ply", "points3d.npy", "trust_scores.npy", "certificates.json", "diagnostics.json"):
        assert (out / name).exists(), name
    certificates = json.loads((out / "certificates.json").read_text())
    assert len(certificates) > 0
    assert any(c["accepted"] for c in certificates)
    trust = np.load(out / "trust_scores.npy")
    assert np.all((trust >= 0) & (trust <= 1))
