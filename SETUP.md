# SETUP.md: environment setup

Tested on Linux with Python 3.13 and R 4.3. Windows works for everything except `make`
(run the commands listed in the Makefile directly).

## 1. Python environment

```bash
git clone --recurse-submodules https://github.com/gabpogi25/capri-stf && cd capri-stf
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Optional components (see the comments in `requirements.txt`):

- **PyTorch**, for training or using `ml_models/`. Pick the CUDA wheel for your GPU at https://pytorch.org/get-started/locally/ and check it with `python -c "import torch; print(torch.cuda.is_available())"`.
- **pycolmap, hloc, trimesh, streamlit, plotly**, for `colmap_capri_automator.py`, `run_real_world_baselines.py` and `dashboard.py`.

## 2. C++ backend (optional)

```bash
make cpp     # = cd cpp_core && python setup.py build_ext --inplace
```

`CAPRI_BACKEND=auto` (the default) uses C++ when the module is built and Python otherwise. `cpp` fails if the module is missing, and `python` forces the reference implementation. Both backends give identical outputs (`runtime.csv`, column `Outputs_Agree`). The greedy selector samples differently in each backend, so its exact tuple choice depends on the backend; every CSV header records which one was used.

## 3. Check the installation

```bash
make test        # fast tests (~20 s)
make test-all    # including slow multi-view/video tests (~1-2 min)
make symbolic    # exact algebraic checks (~30 s)
```

## 4. Third-party code

```bash
git submodule update --init      # SuperGlue (matching) and DUSt3R (run_real_images.py)
python scripts/download_dust3r_weights.py   # only for run_real_images.py
```

COLMAP is located through `$COLMAP_EXE`, then `colmap/COLMAP-3.9.1-windows-cuda/bin/colmap.exe` (on Windows, `setup_env.ps1` downloads it there; the binary is not versioned), then `PATH`. On Linux, install it from your package manager or build it.

FFmpeg (`ffprobe`) is needed by the forensic metadata audit (`forensics/eval/audit_metadata.py`,
used by `experiments/forensics/heldout_detection.py`). Install it and put its `bin` folder on PATH
(Windows: `winget install Gyan.FFmpeg`; Linux: the `ffmpeg` package).

## 5. R (figures)

```bash
# Debian/Ubuntu
sudo apt install r-base r-cran-ggplot2 r-cran-dplyr r-cran-tidyr r-cran-patchwork r-cran-readr r-cran-scales
make figures
```

## 6. Training the ML models (optional)

```bash
python ml_models/shape_prior/train.py --epochs 50 --seed 42
python ml_models/confidence_fusion/train_neural.py --epochs 30 --seed 42
```

Both scripts generate their synthetic training data internally from the seed and write weights to `ml_models/weights/` and curves to `experiments/results/shape_prior*.csv` / `neural_fusion_*.csv`.

## 7. Data

No datasets are included. See `DATA.md` and `data/manifests/README.md`.
For IMC 2025, set `KAGGLE_USERNAME` / `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`)
and run `python scripts/download_imc_dataset.py`.
