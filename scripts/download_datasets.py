"""
scripts/download_datasets.py

Lists the candidate datasets for the forensic study and where to obtain
them. It deliberately downloads nothing automatically: every dataset below
has its own licence and terms, several require registration, and file names
on the hosting pages change. Download manually, check the licence, place the
files as described in data/manifests/README.md, and build a manifest with
scripts/build_manifest.py (which records a sha256 per file).

Guessed file names and small samples of unofficial mirrors would not give a
usable or verifiable dataset, so the script does not fetch them.

Usage:
  python scripts/download_datasets.py            # print the list
  python scripts/download_datasets.py --check    # report which local folders exist
"""

import argparse
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DATASETS = [
    {"name": "Synthbuster", "media": "image", "role": "generated images (9 generators) paired with RAISE prompts",
     "where": "Zenodo record 10066460 (https://zenodo.org/records/10066460)", "local": "images/generated"},
    {"name": "RAISE", "media": "image", "role": "authentic camera RAW/TIFF images (Synthbuster's real side)",
     "where": "http://loki.disi.unitn.it/RAISE/ (registration required)", "local": "images/real"},
    {"name": "GenImage", "media": "image", "role": "generated images from 8 generators + ImageNet reals",
     "where": "https://github.com/GenImage-Dataset/GenImage", "local": "images/generated"},
    {"name": "GenVideo (DeMamba)", "media": "video", "role": "generated and real video clips",
     "where": "https://github.com/chenhaoxing/DeMamba", "local": "videos/generated"},
    {"name": "Own authentic captures", "media": "image/video",
     "role": "phone/camera captures of man-made scenes (best for the structural null)",
     "where": "capture yourself; record device and settings", "local": "images/real, videos/real"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for d in DATASETS:
        print(f"- {d['name']} [{d['media']}]: {d['role']}\n    source: {d['where']}\n    place under: data/{d['local']}/")
    print("\nVerify each licence before use. Datasets are not redistributed with this repository.")
    if args.check:
        for sub in ("images/real", "images/generated", "videos/real", "videos/generated"):
            folder = DATA_ROOT / sub
            n = sum(1 for p in folder.rglob("*") if p.is_file()) if folder.exists() else 0
            print(f"  data/{sub}: {'missing' if not folder.exists() else f'{n} files'}")


if __name__ == "__main__":
    main()
