"""
scripts/build_manifest.py

Build a dataset manifest (data/manifests/README.md) from a folder layout:

    <root>/real/<source>/<file>
    <root>/generated/<generator>/<source>/<file>

Each <source> (scene, prompt or source video) is assigned to ONE split by a
seeded hash, so no source leaks across splits. Generators listed in
--held-out-generators go only to the test split (the held-out detection gates requirement);
the remaining generated items are assigned to cal. Authentic sources are
split between train_null / val / cal / test by --fractions.

Usage:
  python scripts/build_manifest.py --root data/images --out data/manifests/images.csv \
      --held-out-generators sd15 midjourney_v5
"""

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forensics.io.manifests import load_manifest, write_manifest  # noqa: E402

MEDIA = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".mp4", ".mov", ".mkv", ".webm", ".avi"}
SPLITS = ["train_null", "val", "cal", "test"]


def assign(source: str, fractions: list[float], salt: str) -> str:
    u = int(hashlib.sha256(f"{salt}:{source}".encode()).hexdigest()[:12], 16) / 16 ** 12
    edges = [sum(fractions[: i + 1]) for i in range(len(fractions))]
    return SPLITS[next(i for i, e in enumerate(edges) if u < e or i == len(edges) - 1)]


def files(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in MEDIA)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--held-out-generators", nargs="*", default=[])
    parser.add_argument("--fractions", nargs=4, type=float, default=[0.4, 0.2, 0.2, 0.2],
                        help="authentic fractions for train_null val cal test")
    parser.add_argument("--salt", default="capri-forensics-v1")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for source_dir in sorted(p for p in (root / "real").glob("*") if p.is_dir()):
        split = assign(f"real/{source_dir.name}", args.fractions, args.salt)
        rows += [{"path": relative(f), "label": "real", "source": f"real/{source_dir.name}",
                  "generator": "none", "split": split} for f in files(source_dir)]
    for generator_dir in sorted(p for p in (root / "generated").glob("*") if p.is_dir()):
        split = "test" if generator_dir.name in args.held_out_generators else "cal"
        for source_dir in sorted(p for p in generator_dir.glob("*") if p.is_dir()):
            source = f"{generator_dir.name}/{source_dir.name}"
            rows += [{"path": relative(f), "label": "generated", "source": source,
                      "generator": generator_dir.name, "split": split} for f in files(source_dir)]
    if not rows:
        raise SystemExit(f"No media found under {root}/real/<source>/ or {root}/generated/<generator>/<source>/")
    write_manifest(rows, args.out)
    df = load_manifest(args.out)
    print(df.groupby(["split", "label", "generator"]).size().to_string())


if __name__ == "__main__":
    main()
