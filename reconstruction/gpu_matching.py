"""
reconstruction/gpu_matching.py

Exhaustive SuperGlue matching with every feature of a scene held on the GPU.

Same model, configuration, inputs and output format as hloc's match_features
(`matches0` as int16, `matching_scores0` as float16, one group per pair named by
hloc's names_to_pair). What changes is the data handling: hloc reopens the
feature file twice per pair and its writer threads reopen the match file in
append mode, which leaves the GPU idle most of the time. Here the features are
read once, moved to the device once, and the matches are written in one pass.
tests/test_gpu_matching.py checks that the matches equal hloc's.
"""

from __future__ import annotations
import time
from pathlib import Path

import h5py
import numpy as np
import torch


def load_features(feature_path: Path, device: str) -> dict[str, dict[str, torch.Tensor]]:
    """Every image's features from an hloc feature file, as float32 tensors on `device`."""
    features = {}
    with h5py.File(feature_path, "r") as fd:
        for name in fd.keys():
            group = fd[name]
            entry = {k: torch.from_numpy(v.__array__()).float().to(device) for k, v in group.items()}
            # hloc passes an empty image tensor whose shape carries the size
            entry["image"] = torch.empty((1,) + tuple(group["image_size"])[::-1])
            features[name] = entry
    assert features, f"no features in {feature_path}"
    return features


def unique_pairs(pairs_path: Path) -> list[tuple[str, str]]:
    """Pairs from an hloc pairs file, each unordered pair once (first orientation kept)."""
    seen, pairs = set(), []
    for line in Path(pairs_path).read_text().splitlines():
        if not line.strip():
            continue
        name0, name1 = line.split()
        if (name1, name0) in seen or (name0, name1) in seen:
            continue
        seen.add((name0, name1))
        pairs.append((name0, name1))
    return pairs


@torch.no_grad()
def match_scene(conf: dict, pairs_path: Path, feature_path: Path, match_path: Path) -> dict:
    """Match every pair on the GPU (CPU if none); write an hloc-format match file."""
    from hloc import matchers
    from hloc.utils.base_model import dynamic_load
    from hloc.utils.parsers import names_to_pair

    # device_count, not is_available: with the GPU hidden (CUDA_VISIBLE_DEVICES=-1) on Windows
    # is_available() can still be True while no device exists
    device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    t0 = time.perf_counter()
    model = dynamic_load(matchers, conf["model"]["name"])(conf["model"]).eval().to(device)
    features = load_features(feature_path, device)
    pairs = unique_pairs(pairs_path)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = []
    for name0, name1 in pairs:
        # batch dimension of 1 on every entry, as hloc's DataLoader (batch_size=1) adds
        data = {f"{k}0": v.unsqueeze(0) for k, v in features[name0].items()}
        data.update({f"{k}1": v.unsqueeze(0) for k, v in features[name1].items()})
        pred = model(data)
        results.append((names_to_pair(name0, name1),
                        pred["matches0"][0].short().cpu().numpy(),
                        pred["matching_scores0"][0].half().cpu().numpy()))
    if device == "cuda":
        torch.cuda.synchronize()
    t_match = time.perf_counter() - t0

    match_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(match_path, "w", libver="latest") as fd:
        for pair, matches0, scores0 in results:
            group = fd.create_group(pair)
            group.create_dataset("matches0", data=matches0)
            group.create_dataset("matching_scores0", data=scores0)
    return {"device": device, "pairs": len(pairs), "load_s": t_load, "match_s": t_match,
            "pairs_per_s": len(pairs) / max(t_match, 1e-9)}


def same_matches(path_a: Path, path_b: Path) -> tuple[int, int]:
    """(pairs compared, pairs whose matches0 differ) between two hloc match files."""
    differing = 0
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        names_a = _pair_names(fa)
        names_b = _pair_names(fb)
        assert names_a == names_b, "match files cover different pairs"
        for name in names_a:
            if not np.array_equal(_dataset(fa, name, "matches0"), _dataset(fb, name, "matches0")):
                differing += 1
    return len(names_a), differing


def _pair_names(fd: h5py.File) -> list[str]:
    names = []
    fd.visit(lambda n: names.append(n.rsplit("/", 1)[0]) if n.endswith("/matches0") else None)
    return sorted(names)


def _dataset(fd: h5py.File, pair: str, key: str) -> np.ndarray:
    return fd[f"{pair}/{key}"][()]
