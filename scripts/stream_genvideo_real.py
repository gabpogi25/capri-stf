"""
scripts/stream_genvideo_real.py

Real videos for the forensic study from GenVideo-100K (ModelScope cccnju/GenVideo-100K): the
archive Real_part_* is one gzip-compressed tar split into parts. This script streams the
FIRST part (Real_part_aa, 30 GB) through gzip and tar, writes the video members as they arrive
to data/videos_youku/real/<clip>/<file> (the layout of scripts/build_manifest.py), and stops
after --max_files clips or --max_gb gigabytes, whichever comes first. The archive itself is
never stored.

Why: the real side of GenVideo-Val (MSR-VTT previews, 298x224 at 3 fps) differs from its
generated side in resolution, frame rate and length, so no fair test is possible on it.

Integrity: the published sha256 covers the whole 30 GB part and cannot be checked on a partial
stream. Each written file's sha256 is recorded by build_manifest.py, and every clip is decoded
once here; a clip that does not decode is deleted and counted.

Usage:
  python scripts/stream_genvideo_real.py --max_files 4000 --max_gb 30
"""

from __future__ import annotations
import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
TARGET = DATA_ROOT / "videos_youku" / "real"
URL = "https://modelscope.cn/api/v1/datasets/cccnju/GenVideo-100K/repo?Revision=master&FilePath=Real_part_aa"
USER_AGENT = "capri-forensic-fetch/1.0 (research download)"
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
RESERVE_GB = 20.0


def decodes(path: Path) -> bool:
    import cv2

    capture = cv2.VideoCapture(str(path))
    ok, _ = capture.read()
    capture.release()
    return bool(ok)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_files", type=int, default=4000)
    parser.add_argument("--max_gb", type=float, default=30.0)
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(TARGET).free / 1e9
    if free_gb < args.max_gb + RESERVE_GB:
        sys.exit(f"Refusing: {free_gb:.1f} GB free, {args.max_gb + RESERVE_GB:.1f} GB needed")

    written = existing = undecodable = 0
    current: Path | None = None   # file being written; deleted if the stream stops inside it
    total_bytes, t0 = 0, time.monotonic()
    with requests.get(URL, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        archive = tarfile.open(fileobj=response.raw, mode="r|gz")
        try:
            for member in archive:
                if not member.isfile() or Path(member.name).suffix.lower() not in VIDEO:
                    continue
                name = Path(member.name).name
                target = TARGET / Path(name).stem / name
                if target.exists() and target.stat().st_size == member.size:
                    existing += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    current = target
                    with archive.extractfile(member) as source, open(target, "wb") as out:
                        shutil.copyfileobj(source, out, 8 * 1024 * 1024)
                    current = None
                    if not decodes(target):
                        target.unlink()
                        target.parent.rmdir()
                        undecodable += 1
                        continue
                    written += 1
                total_bytes += member.size
                if (written + existing) % 200 == 0:
                    rate = total_bytes / max(time.monotonic() - t0, 1e-9) / 1e6
                    print(f"{written + existing} clips, {total_bytes / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)
                if written + existing >= args.max_files or total_bytes / 1e9 >= args.max_gb:
                    break
        except (EOFError, tarfile.ReadError, requests.exceptions.ChunkedEncodingError) as error:
            if current is not None and current.exists():
                current.unlink()
            print(f"stream ended inside the split archive ({error}); clips so far are complete", flush=True)
    print(f"done: {written} written, {existing} already present, {undecodable} undecodable deleted, "
          f"{total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
