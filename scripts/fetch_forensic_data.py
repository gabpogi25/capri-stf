"""
scripts/fetch_forensic_data.py

Downloads the openly available forensic datasets with resume, verifies each file's size and
published checksum, and refuses to start without enough free disk space. Files go to
data/raw/<dataset>/ (data/ is git-ignored); arranging them into the layout of
data/manifests/README.md is a separate step (--place), written once the archive contents
are known, because the archives' internal layout is not published.

Never stores credentials. Registration-gated datasets (RAISE, FaceForensics++) are not
fetched here: they are listed by --list with the manual step each needs.

Usage:
  python scripts/fetch_forensic_data.py --list
  python scripts/fetch_forensic_data.py --fetch synthbuster genvideo_val
  python scripts/fetch_forensic_data.py --verify synthbuster genvideo_val
"""

from __future__ import annotations
import argparse
import hashlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
RAW_ROOT = DATA_ROOT / "raw"
USER_AGENT = "capri-forensic-fetch/1.0 (research download; resumable)"
CHUNK = 8 * 1024 * 1024
# Free space required before a download starts: what is still to download, the same again
# for extracting it, and a fixed reserve for the reconstruction outputs and the system.
EXTRACT_FACTOR = 2.0
RESERVE_GB = 20.0


@dataclass(frozen=True)
class Source:
    name: str
    filename: str
    url: str
    size: int            # bytes, as published by the host
    algorithm: str       # "md5" or "sha256"
    checksum: str        # as published by the host
    licence: str
    role: str


SOURCES = {
    "synthbuster": Source(
        name="synthbuster", filename="synthbuster.zip",
        url="https://zenodo.org/records/10066460/files/synthbuster.zip?download=1",
        size=12_370_000_000, algorithm="md5", checksum="0695bd328e16ea21c5c9cc2ae1d994ff",
        licence="CC BY-NC-SA 4.0 (Zenodo record 10066460)",
        role="generated images, 9 generators, prompts paired with RAISE-1k (reals: RAISE, gated)"),
    "genvideo_val": Source(
        name="genvideo_val", filename="GenVideo-Val.zip",
        url="https://modelscope.cn/api/v1/datasets/cccnju/Gen-Video/repo?Revision=master&FilePath=GenVideo-Val.zip",
        size=13_940_000_000, algorithm="sha256",
        checksum="aba02ecd57528257249dd38b72b7a34c9d56bc75d4f2541a721b2778f1d864a6",
        licence="not stated on the DeMamba/GenVideo pages; check before any redistribution",
        role="GenVideo test set: real MSR-VTT clips and generated clips from many generators"),
}

GATED = {
    "raise": "RAISE (real side of Synthbuster): register at http://loki.disi.unitn.it/RAISE/, then "
             "download the RAISE-1k list; ~1,000 RAW/TIFF images, tens of GB.",
    "ffpp": "FaceForensics++: submit the access form linked from https://github.com/ondyari/FaceForensics; "
            "the download script arrives by e-mail after approval.",
}


def published_size(source: Source) -> int:
    """Exact size from the host (Content-Range of a 1-byte request); falls back to the table."""
    response = requests.get(source.url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
                            stream=True, timeout=60, allow_redirects=True)
    response.close()
    content_range = response.headers.get("Content-Range", "")
    return int(content_range.rsplit("/", 1)[1]) if "/" in content_range else source.size


def check_disk(target: Path, remaining_bytes: int) -> None:
    """Refuse to start unless free space covers download + extraction + reserve."""
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / 1e9
    needed_gb = remaining_bytes / 1e9 * EXTRACT_FACTOR + RESERVE_GB
    if free_gb < needed_gb:
        sys.exit(f"Refusing to download: {free_gb:.1f} GB free on {target.anchor}, "
                 f"{needed_gb:.1f} GB needed ({EXTRACT_FACTOR:g}x the remaining "
                 f"{remaining_bytes / 1e9:.1f} GB plus a {RESERVE_GB:g} GB reserve).")
    print(f"Disk check: {free_gb:.1f} GB free, {needed_gb:.1f} GB needed.")


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: Source, size: int) -> Path:
    """Download with resume into data/raw/<name>/<file>.part, then rename when complete."""
    folder = RAW_ROOT / source.name
    folder.mkdir(parents=True, exist_ok=True)
    final, part = folder / source.filename, folder / (source.filename + ".part")
    if final.exists() and final.stat().st_size == size:
        print(f"{source.filename}: already complete")
        return final
    start = part.stat().st_size if part.exists() else 0
    assert start <= size, f"{part} is larger than the published size; delete it"
    t0, done = time.monotonic(), start
    while done < size:
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={done}-"}
        try:
            with requests.get(source.url, headers=headers, stream=True, timeout=120) as response:
                assert response.status_code == 206, f"server ignored the range request ({response.status_code})"
                with open(part, "ab") as f:
                    for block in response.iter_content(CHUNK):
                        f.write(block)
                        done += len(block)
                        if done % (512 * 1024 * 1024) < CHUNK:
                            rate = (done - start) / max(time.monotonic() - t0, 1e-9) / 1e6
                            print(f"{source.filename}: {done / 1e9:.2f} / {size / 1e9:.2f} GB, {rate:.1f} MB/s", flush=True)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as error:
            print(f"{source.filename}: connection dropped at {done / 1e9:.2f} GB ({error}); resuming in 30 s", flush=True)
            time.sleep(30)
            done = part.stat().st_size
    assert part.stat().st_size == size, f"{part}: {part.stat().st_size} bytes, expected {size}"
    part.rename(final)
    return final


def verify(source: Source, path: Path) -> None:
    """Compare the file's checksum with the published one; a mismatch stops the script."""
    print(f"{source.filename}: computing {source.algorithm} ...", flush=True)
    actual = file_digest(path, source.algorithm)
    if actual != source.checksum:
        sys.exit(f"{source.filename}: {source.algorithm} mismatch ({actual}); the file is kept for inspection")
    print(f"{source.filename}: {source.algorithm} OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--fetch", nargs="+", choices=sorted(SOURCES))
    parser.add_argument("--verify", nargs="+", choices=sorted(SOURCES))
    args = parser.parse_args()

    if args.list or not (args.fetch or args.verify):
        for s in SOURCES.values():
            print(f"- {s.name}: {s.filename}, about {s.size / 1e9:.1f} GB, {s.algorithm} published\n"
                  f"    {s.role}\n    licence: {s.licence}")
        for name, step in GATED.items():
            print(f"- {name} (manual step first): {step}")
        return

    for name in args.verify or []:
        source = SOURCES[name]
        verify(source, RAW_ROOT / name / source.filename)

    if args.fetch:
        sizes = {name: published_size(SOURCES[name]) for name in args.fetch}
        remaining = 0
        for name, size in sizes.items():
            folder = RAW_ROOT / name
            have = sum(p.stat().st_size for p in folder.glob(SOURCES[name].filename + "*")) if folder.exists() else 0
            remaining += max(size - have, 0)
        check_disk(RAW_ROOT, remaining)
        for name in args.fetch:
            path = download(SOURCES[name], sizes[name])
            verify(SOURCES[name], path)


if __name__ == "__main__":
    main()
