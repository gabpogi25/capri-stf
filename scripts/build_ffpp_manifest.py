"""
scripts/build_ffpp_manifest.py

Manifest for the FaceForensics++ forensic study (docs/prereg/forensics_ffpp.md) from the files
downloaded by FaceForensics' own script (data/raw/ffpp/, c23, see the pre-registration).

FF++ is built from pairs of original videos (A, B): the manipulated videos A_B (target A, face
of B) and B_A use both. So the unit of the split is the PAIR: both originals and every
manipulated video made from them go to one split, decided by a salted hash of the pair, with the
fractions of scripts/build_manifest.py (train_null 0.4, val 0.2, cal 0.2, test 0.2). A face used
to fit the null can then never reappear in a test video. All four manipulation methods are held
out: every manipulated video takes its pair's split; only those in test pairs are read (gate A
reads authentic items only, gate B the test split and authentic val items). source = the pair, so cluster bootstraps resample pairs.

Files are referenced where the download script put them (no copy).

Usage:
  python scripts/build_ffpp_manifest.py --out data/manifests/ffpp_c23.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from build_manifest import assign, relative  # noqa: E402
from forensics.io.manifests import load_manifest, write_manifest  # noqa: E402

RAW = ROOT / "data" / "raw" / "ffpp"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FRACTIONS = [0.4, 0.2, 0.2, 0.2]
SALT = "capri-forensics-ffpp-v1"


def pair_key(a: str, b: str) -> str:
    return "pair_" + "_".join(sorted((a, b)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compression", default="c23")
    args = parser.parse_args()
    originals = sorted((RAW / "original_sequences" / "youtube" / args.compression / "videos").glob("*.mp4"))
    fakes = {m: sorted((RAW / "manipulated_sequences" / m / args.compression / "videos").glob("*.mp4"))
             for m in METHODS}
    have = {p.stem for p in originals}
    pair_of: dict[str, str] = {}
    for paths in fakes.values():
        for p in paths:
            a, b = p.stem.split("_")
            assert a in have and b in have, f"{p.name}: an original of its pair is missing"
            key = pair_key(a, b)
            for v in (a, b):
                assert pair_of.setdefault(v, key) == key, f"original {v} is in two pairs"
    rows = []
    for p in originals:
        key = pair_of.get(p.stem)
        assert key is not None, f"original {p.stem} has no manipulated video"
        rows.append({"path": relative(p), "label": "real", "source": key, "generator": "none",
                     "split": assign(key, FRACTIONS, SALT)})
    for method, paths in fakes.items():
        for p in paths:
            key = pair_key(*p.stem.split("_"))
            split = assign(key, FRACTIONS, SALT)   # the pair's split, so no pair spans two splits
            rows.append({"path": relative(p), "label": "generated", "source": key,
                         "generator": method.lower(), "split": split})
    write_manifest(rows, args.out)
    df = load_manifest(args.out)
    print(df.groupby(["split", "label", "generator"]).size().to_string())
    print("pairs per split:", df[df.label == "real"].groupby("split")["source"].nunique().to_dict())


if __name__ == "__main__":
    main()
