"""Parallel match filtering and GPU matching give the same output as their serial references."""

from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reconstruction.gpu_matching import same_matches, unique_pairs  # noqa: E402
from reconstruction.match_filtering import filter_matches  # noqa: E402


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    x = P @ np.vstack([X, np.ones(X.shape[1])])
    return (x[:2] / x[2]).T


def _write_scene(tmp: Path, n_images: int = 3, n_points: int = 120, seed: int = 7) -> tuple[Path, Path]:
    """hloc-format feature and match files for a synthetic scene with 10% wrong matches."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(3, n_points)) + np.array([[0], [0], [6]])
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    names = [f"im{i}.png" for i in range(n_images)]
    keypoints = {}
    for i, name in enumerate(names):
        angle = 0.08 * i
        R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        P = K @ np.hstack([R, np.array([[0.4 * i], [0], [0]])])
        keypoints[name] = (_project(P, X) + rng.normal(0, 0.5, size=(n_points, 2))).astype(np.float32)
    features, matches = tmp / "features.h5", tmp / "matches.h5"
    with h5py.File(features, "w") as fd:
        for name in names:
            fd.create_group(name).create_dataset("keypoints", data=keypoints[name])
    with h5py.File(matches, "w") as fd:
        for i, name0 in enumerate(names):
            group = fd.create_group(name0)
            for name1 in names[i + 1:]:
                matches0 = np.arange(n_points, dtype=np.int16)
                wrong = rng.choice(n_points, size=n_points // 10, replace=False)
                matches0[wrong] = rng.permutation(matches0[wrong])
                pair = group.create_group(name1)
                pair.create_dataset("matches0", data=matches0)
                pair.create_dataset("matching_scores0", data=np.ones(n_points, dtype=np.float16))
    return features, matches


@pytest.mark.parametrize("mode", ["ransac", "ransac_capri"])
def test_parallel_filtering_equals_serial(tmp_path: Path, mode: str) -> None:
    features, matches = _write_scene(tmp_path)
    serial = filter_matches(features, matches, tmp_path / "serial.h5", mode, seed=11, n_workers=1)
    parallel = filter_matches(features, matches, tmp_path / "parallel.h5", mode, seed=11, n_workers=3)
    assert same_matches(tmp_path / "serial.h5", tmp_path / "parallel.h5") == (3, 0)
    for key in ("matches_after_ransac", "matches_out", "pairs_capri_empty"):
        assert serial[key] == parallel[key]
    assert serial["matches_out"] < serial["matches_in"]  # the wrong matches are removed


def test_opencv_error_is_a_pair_without_model(tmp_path: Path, monkeypatch) -> None:
    import cv2

    features, matches = _write_scene(tmp_path)

    def fail(*args, **kwargs):
        raise cv2.error("(-215:Assertion failed) !model.empty()")

    monkeypatch.setattr(cv2, "findFundamentalMat", fail)
    stats = filter_matches(features, matches, tmp_path / "out.h5", "ransac_capri", seed=11, n_workers=1)
    assert stats["pairs_ransac_error"] == 3 and stats["matches_out"] == 0
    assert stats["pairs_capri_empty"] == 0


def test_mode_none_keeps_every_match(tmp_path: Path) -> None:
    features, matches = _write_scene(tmp_path)
    stats = filter_matches(features, matches, tmp_path / "none.h5", "none", seed=11)
    assert stats["matches_out"] == stats["matches_in"]
    assert same_matches(tmp_path / "none.h5", matches) == (3, 0)


def test_unique_pairs_keeps_first_orientation(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text("a b\nb a\na c\nc b\n")
    assert unique_pairs(pairs_path) == [("a", "b"), ("a", "c"), ("c", "b")]


def test_gpu_matching_equals_hloc(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("hloc")
    from hloc import match_features
    from reconstruction.gpu_matching import match_scene

    rng = np.random.default_rng(3)
    names = ["a.png", "b.png", "c.png"]
    base = rng.normal(size=(128, 60)).astype(np.float32)
    features = tmp_path / "features.h5"
    with h5py.File(features, "w") as fd:
        for name in names:
            group = fd.create_group(name)
            group.create_dataset("descriptors", data=base + 0.05 * rng.normal(size=base.shape).astype(np.float32))
            group.create_dataset("keypoints", data=rng.uniform(0, 100, size=(60, 2)).astype(np.float32))
            group.create_dataset("image_size", data=np.array([100, 100]))
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.png b.png\na.png c.png\nb.png c.png\n")
    conf = match_features.confs["NN-ratio"]
    match_scene(conf, pairs, features, tmp_path / "ours.h5")
    from colmap_capri_automator import single_process_loading

    with single_process_loading():  # hloc's loader workers hang on Windows
        match_features.main(conf, pairs, features=features, matches=tmp_path / "hloc.h5")
    assert same_matches(tmp_path / "ours.h5", tmp_path / "hloc.h5") == (3, 0)
    assert torch is not None


def test_video_usac_errors_count_as_no_model(monkeypatch) -> None:
    import cv2
    from forensics.video import multiview, tracking

    def fail(*args, **kwargs):
        raise cv2.error("(-215:Assertion failed) !model.empty()")

    monkeypatch.setattr(cv2, "findFundamentalMat", fail)
    monkeypatch.setattr(cv2, "findHomography", fail)
    tracks = np.random.default_rng(0).uniform(0, 100, size=(30, 6, 2))
    assert not tracking.rigid_inliers(tracks, 1.0, 0.999, 0.8).any()
    ok, info = tracking.parallax_ok(tracks, 640, 480, 0.02, 1.2, 1.0)
    assert not ok and info["inliers_f"] == 0 and info["inliers_h"] == 0
    with pytest.raises(RuntimeError):
        multiview.projective_reconstruction([tracks[:, v] / 100 for v in range(3)])
