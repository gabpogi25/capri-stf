"""
scripts/place_forensic_data.py

Arranges the downloaded archives (scripts/fetch_forensic_data.py, data/raw/) into the layout
read by scripts/build_manifest.py (data/manifests/README.md):

    data/videos/real/<source>/<file>              GenVideo-Val/Real/*            (MSR-VTT)
    data/videos/generated/<generator>/<source>/<file>  GenVideo-Val/Fake/<generator>/*
    data/images/generated/<generator>/<source>/<file>  synthbuster/<generator>/*.png

<source> is the file name without its extension: one clip or one prompt per source. Files are
streamed from the archive (zipfile checks each member's CRC while reading), so nothing is
extracted twice. macOS metadata (__MACOSX, .DS_Store) and formats build_manifest.py does not
read (e.g. GIF) are skipped and counted. Synthbuster's real side is RAISE (registration
required) and is placed separately.

Usage:
  python scripts/place_forensic_data.py genvideo_val synthbuster
"""

from __future__ import annotations
import argparse
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_ROOT = DATA_ROOT / "raw"
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
RESERVE_GB = 20.0


def target_genvideo(member: str) -> Path | None:
    parts = member.split("/")
    if len(parts) == 3 and parts[:2] == ["GenVideo-Val", "Real"]:
        name = parts[2]
        return DATA_ROOT / "videos" / "real" / Path(name).stem / name
    if len(parts) == 4 and parts[:2] == ["GenVideo-Val", "Fake"]:
        generator, name = parts[2].lower(), parts[3]
        return DATA_ROOT / "videos" / "generated" / generator / Path(name).stem / name
    return None


def target_synthbuster(member: str) -> Path | None:
    parts = member.split("/")
    if len(parts) == 3 and parts[0] == "synthbuster":
        generator, name = parts[1], parts[2]
        return DATA_ROOT / "images" / "generated" / generator / Path(name).stem / name
    return None


ARCHIVES = {
    "genvideo_val": (RAW_ROOT / "genvideo_val" / "GenVideo-Val.zip", target_genvideo, VIDEO),
    "synthbuster": (RAW_ROOT / "synthbuster" / "synthbuster.zip", target_synthbuster, IMAGE),
}


def place(name: str) -> Counter:
    archive, target_of, media = ARCHIVES[name]
    assert archive.exists(), f"{archive} missing; run scripts/fetch_forensic_data.py --fetch {name}"
    counts = Counter()
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        needed = sum(m.file_size for m in members) / 1e9
        free = shutil.disk_usage(DATA_ROOT).free / 1e9
        if free < needed + RESERVE_GB:
            sys.exit(f"Refusing: {free:.1f} GB free, {needed + RESERVE_GB:.1f} GB needed for {name}")
        for member in members:
            path = member.filename
            if path.startswith("__MACOSX/") or Path(path).name.startswith("."):
                counts["skipped: macOS metadata"] += 1
                continue
            target = target_of(path)
            if target is None or target.suffix.lower() not in media:
                counts[f"skipped: {Path(path).suffix.lower() or 'no extension'}"] += 1
                continue
            if target.exists() and target.stat().st_size == member.file_size:
                counts["already placed"] += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, open(target, "wb") as out:
                shutil.copyfileobj(source, out, 8 * 1024 * 1024)
            counts[f"placed: {target.relative_to(DATA_ROOT).parts[0]}/{target.relative_to(DATA_ROOT).parts[1]}"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", choices=sorted(ARCHIVES))
    args = parser.parse_args()
    for name in args.archives:
        print(f"== {name}")
        for key, value in sorted(place(name).items()):
            print(f"  {value:6d}  {key}")


if __name__ == "__main__":
    main()
