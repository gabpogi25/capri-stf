import os
import shutil
import time
from pathlib import Path

def main():
    sg_dir = Path("outputs/benchmark_real/superglue")
    capri_dir = Path("outputs/benchmark_real/capri")
    capri_dir.mkdir(parents=True, exist_ok=True)
    
    print("Waiting for SuperGlue to finish matching (indicated by sparse folder creation)...")
    while not (sg_dir / "sparse" / "models").exists() and not (sg_dir / "sparse" / "0").exists():
        time.sleep(10)
        
    print("SuperGlue matches finished! Copying to CAPRI folder to save 3.5 hours...")
    
    # Wait an extra 10 seconds just to ensure the file handles are closed
    time.sleep(10)
    
    # Copy features
    if (sg_dir / "features.h5").exists():
        shutil.copy(sg_dir / "features.h5", capri_dir / "features.h5")
        print("Copied features.h5")
        
    # Copy matches
    if (sg_dir / "matches.h5").exists():
        shutil.copy(sg_dir / "matches.h5", capri_dir / "matches.h5")
        print("Copied matches.h5")
        
    print("Done! CAPRI phase will now instantly skip extraction and matching.")

if __name__ == "__main__":
    main()
