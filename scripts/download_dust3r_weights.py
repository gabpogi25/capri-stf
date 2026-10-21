"""
scripts/download_dust3r_weights.py

Downloads the pretrained weights required for DUSt3R (ViT-Large BaseDecoder).
These weights are hosted by Naver Labs.

Weights size: ~2.4 GB
"""

import os
import sys
import urllib.request
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUST3R_DIR = PROJECT_ROOT / "dust3r"
CHECKPOINT_DIR = DUST3R_DIR / "checkpoints"


def download_with_progress(url, dest_path):
    import time
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r  Downloaded: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print("\n")


def download_weights():
    print("=" * 60)
    print("DUSt3R PRETRAINED WEIGHT DOWNLOADER")
    print("=" * 60)

    if not DUST3R_DIR.exists():
        print(f"ERROR: DUSt3R directory not found at {DUST3R_DIR}")
        print("Please ensure DUSt3R is cloned into the project root.")
        sys.exit(1)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # URL for ViT-Large BaseDecoder (resolution 512)
    # Standard checkpoint used in DUSt3R paper
    url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    file_name = url.split("/")[-1]
    dest_path = CHECKPOINT_DIR / file_name

    if dest_path.exists():
        print(f"Weights already exist at: {dest_path}")
        print("Skipping download.")
        return

    print(f"\nDownloading {file_name} (~2.4 GB)...")
    print(f"Saving to: {dest_path}\n")

    try:
        download_with_progress(url, dest_path)
        print(f"Download complete!")
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Remove partial file
        sys.exit(1)


if __name__ == "__main__":
    download_weights()
