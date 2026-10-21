"""
scripts/download_imc_dataset.py

Downloads the Image Matching Challenge 2025 dataset from Kaggle
using the kagglehub API.

Prerequisites:
  1. pip install kagglehub
  2. Kaggle credentials in the environment or ~/.kaggle/kaggle.json

Usage:
  python scripts/download_imc_dataset.py
"""

import os
import sys
from pathlib import Path

# Credentials are read from the environment (KAGGLE_USERNAME/KAGGLE_KEY or
# KAGGLE_API_TOKEN) or ~/.kaggle/kaggle.json. Never hard-code a token here:
# a token committed to git is public to anyone who can read the history.
if not (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
        or (Path.home() / ".kaggle" / "kaggle.json").exists()):
    sys.exit("Kaggle credentials not found. Set KAGGLE_API_TOKEN or create ~/.kaggle/kaggle.json.")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "real_images"


def download_dataset():
    print("=" * 60)
    print("IMC 2025 DATASET DOWNLOADER")
    print("=" * 60)

    # Check kagglehub is installed
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("Run: pip install kagglehub")
        sys.exit(1)

    print(f"\nDownloading IMC 2025 dataset (~5.22 GB)...")
    print(f"This may take a while depending on your internet speed.\n")

    try:
        # Download using kagglehub API
        path = kagglehub.competition_download('image-matching-challenge-2025')
        print(f"\nDownload complete!")
        print(f"Dataset cached at: {path}")

        # Create our data directory and save reference
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        ref_file = DATA_DIR / "imc2025_path.txt"
        with open(ref_file, "w") as f:
            f.write(str(path))
        print(f"Dataset path saved to: {ref_file}")

        # List what we downloaded
        print(f"\nDataset contents:")
        dl_path = Path(path)
        for item in sorted(dl_path.iterdir()):
            if item.is_dir():
                n_files = len(list(item.rglob("*")))
                print(f"  [DIR]  {item.name}/ ({n_files} files)")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  [FILE] {item.name} ({size_mb:.1f} MB)")

        print(f"\nDone! The dataset is ready for use by run_real_images.py")
        return str(path)

    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have accepted the competition rules on Kaggle")
        print("     Visit: https://www.kaggle.com/competitions/image-matching-challenge-2025")
        print("  2. Check your internet connection")
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
