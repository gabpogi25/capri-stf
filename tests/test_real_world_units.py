"""Scene units of the real-image benchmark: one IMC scene per unit, outliers never used."""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import run_real_world_baselines as rwb  # noqa: E402
sys.path.insert(0, str(ROOT / "experiments" / "eval"))
import real_reconstruction_analysis as rra  # noqa: E402
from colmap_capri_automator import keep_largest_model  # noqa: E402

IDENTITY = "1;0;0;0;1;0;0;0;1"


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    rows = [
        ("ds", "a", "a1.png", IDENTITY, "0;0;0"),
        ("ds", "a", "a2.png", IDENTITY, "1;0;0"),
        ("ds", "b", "b1.png", IDENTITY, "0;5;0"),
        ("ds", "outliers", "o1.png", "nan", "nan"),
    ]
    pd.DataFrame(rows, columns=["dataset", "scene", "image", "rotation_matrix", "translation_vector"]) \
        .to_csv(tmp_path / "train_labels.csv", index=False)
    (tmp_path / "train" / "ds").mkdir(parents=True)
    for _, _, name, _, _ in rows:
        (tmp_path / "train" / "ds" / name).write_bytes(b"img")
    return tmp_path


def test_units_exclude_outliers(dataset_dir: Path) -> None:
    assert rwb.list_units(dataset_dir / "train_labels.csv") == ["ds/a", "ds/b"]


def test_parse_unit_rejects_bad_units() -> None:
    assert rwb.parse_unit("ds/a") == ("ds", "a")
    with pytest.raises(AssertionError):
        rwb.parse_unit("ds")
    with pytest.raises(AssertionError):
        rwb.parse_unit("ds/outliers")


def test_staging_copies_only_the_scene(dataset_dir: Path, tmp_path: Path) -> None:
    dest = tmp_path / "staged"
    n = rwb.stage_scene_images(dataset_dir, dataset_dir / "train_labels.csv", "ds", "a", dest)
    assert n == 2
    assert sorted(p.name for p in dest.iterdir()) == ["a1.png", "a2.png"]


def test_ground_truth_is_per_scene(dataset_dir: Path) -> None:
    poses = rwb.load_ground_truth(dataset_dir / "train_labels.csv", "ds", "a")
    assert sorted(poses) == ["a1.png", "a2.png"]
    assert poses["a2.png"]["t"].tolist() == [1.0, 0.0, 0.0]


def test_unregistered_images_count_as_misses(dataset_dir: Path) -> None:
    poses = rwb.load_ground_truth(dataset_dir / "train_labels.csv", "ds", "a")
    poses["a3.png"] = poses["a2.png"]
    estimated = {name: poses[name] for name in ("a1.png", "a2.png")}
    errors = rwb.relative_pose_errors(estimated, poses)
    assert errors.size == 3  # every ground-truth pair, not only the registered one
    assert sorted(errors.tolist())[:1] == [0.0] and int((errors == float("inf")).sum()) == 2


def test_keep_largest_model_copies_root_model(tmp_path: Path) -> None:
    # hloc moves the largest model into the sfm directory and leaves models/<k> empty
    (tmp_path / "models" / "0").mkdir(parents=True)
    for stem in ("cameras", "images", "points3D"):
        (tmp_path / f"{stem}.bin").write_bytes(stem.encode())
    keep_largest_model(tmp_path)
    assert sorted(p.name for p in (tmp_path / "0").iterdir()) == ["cameras.bin", "images.bin", "points3D.bin"]
    assert (tmp_path / "0" / "images.bin").read_bytes() == b"images"


def test_decision_rule_matches_preregistration() -> None:
    assert rra.decide(0.001, 0.05) == "benefit supported"
    assert rra.decide(-0.05, -0.001) == "harm"
    assert rra.decide(-0.005, 0.008) == "no difference shown: equivalent within 0.01 mAA"
    assert rra.decide(-0.02, 0.03) == "no difference shown: inconclusive"


def test_interim_analysis_never_decides() -> None:
    label = rra.interim_label(1)
    assert "no decision" in label and "supported" not in label and "harm" not in label


def test_batch_draw_is_fixed_by_its_seed_and_rule() -> None:
    import draw_real_batches as drb

    sizes = pd.DataFrame({"Dataset": ["d"] * 14, "Scene": [f"s{i}" for i in range(14)],
                          "N_Images": [10, 20, 30, 40, 50, 60, 70, 75, 76, 80, 12, 14, 16, 200]})
    a, b = drb.draw_batches(sizes), drb.draw_batches(sizes)
    assert a["Batch"].tolist() == b["Batch"].tolist()  # same seed, same draw
    assert (a.loc[a["N_Images"] > drb.MAX_IMAGES, "Batch"] == 3).all()
    assert (a["Batch"] == 1).sum() == drb.BATCH1_SIZE
    assert not a.loc[a["N_Images"] <= drb.MAX_IMAGES, "Batch"].eq(3).any()


def test_paired_differences_average_seeds_per_unit() -> None:
    rows = [("d", s, p, seed, v) for s, base in (("a", 0.5), ("b", 0.7))
            for seed in (0, 1) for p, v in (("superglue_ransac", base), ("superglue_ransac_capri", base + 0.1 * seed))]
    df = pd.DataFrame(rows, columns=["Dataset", "Scene", "Pipeline", "Seed", "mAA_at_10"])
    d = rra.paired_differences(df, "mAA_at_10")
    assert d.round(10).tolist() == [0.05, 0.05]
    low, high = rra.bootstrap_ci(d.to_numpy())
    assert low == pytest.approx(0.05) and high == pytest.approx(0.05)


def test_progress_round_trip_and_refuses_other_code(tmp_path: Path, monkeypatch) -> None:
    import forensics.io.provenance as provenance

    path = tmp_path / "progress.jsonl"
    row = {"Dataset": "ds", "Scene": "a", "Pipeline": "sift", "Seed": 1, "mAA_at_10": 0.5,
           "Median_Pose_Error_deg": float("inf")}
    monkeypatch.setattr(provenance, "git_commit", lambda: "abc1234")
    rwb.append_progress(path, row)
    assert rwb.load_progress(path) == {("ds", "a", "sift", 1): row}
    monkeypatch.setattr(provenance, "git_commit", lambda: "def5678")
    with pytest.raises(AssertionError):
        rwb.load_progress(path)


def test_keep_largest_model_without_model_does_nothing(tmp_path: Path) -> None:
    keep_largest_model(tmp_path)
    assert not (tmp_path / "0").exists()
