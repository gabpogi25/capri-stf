# CAPRI reproducibility targets. Run from the repository root.
PYTHON ?= python
RSCRIPT ?= Rscript

.PHONY: all install cpp test test-all symbolic experiments figures abstracts paper-figures forensics-fixtures clean-cache

all: cpp test experiments figures

install:
	$(PYTHON) -m pip install -r requirements.txt

cpp:                       ## build the C++ backend (optional; Python fallback otherwise)
	cd cpp_core && $(PYTHON) setup.py build_ext --inplace

test:                      ## fast tests
	$(PYTHON) -m pytest -q -m "not slow"

test-all:                  ## all tests including slow multi-view/video tests
	$(PYTHON) -m pytest -q

symbolic:                  ## exact checks of the Plucker relations and strata codimensions (SymPy; Macaulay2 if installed)
	{ echo "# $$($(PYTHON) --version), sympy $$($(PYTHON) -c 'import sympy; print(sympy.__version__)'); commit $$(git rev-parse --short HEAD 2>/dev/null || echo unknown); command: python experiments/symbolic/verify_plucker.py"; \
	  $(PYTHON) experiments/symbolic/verify_plucker.py; } > experiments/symbolic/verify_plucker_output.txt
	cat experiments/symbolic/verify_plucker_output.txt
	@if command -v M2 >/dev/null 2>&1; then \
	  { echo "# Macaulay2 $$(M2 --version); command: M2 --script experiments/symbolic/verify_plucker.m2"; \
	    M2 --script experiments/symbolic/verify_plucker.m2; } > experiments/symbolic/verify_plucker_m2_output.txt; \
	  cat experiments/symbolic/verify_plucker_m2_output.txt; \
	else echo "Macaulay2 (M2) not installed: verify_plucker.m2 skipped"; fi

experiments:               ## every synthetic 3D-reconstruction experiment -> experiments/results/*.csv
	CAPRI_BACKEND=auto $(PYTHON) experiments/run_all.py

forensics-fixtures:        ## forensic implementation checks on synthetic fixtures (synthetic null, multi-view and video fixtures)
	cd experiments/forensics && $(PYTHON) synthetic_null.py && $(PYTHON) multiview_synthetic.py && $(PYTHON) video_fixture.py

figures:                   ## all figures and summary tables -> figures/
	$(RSCRIPT) figures/make_figures.R

abstracts:                 ## graphical abstracts (R assets + Inkscape) -> figures/abstracts/*.pdf|png|svg
	bash figures/abstracts/build.sh

paper-figures: abstracts   ## illustrative paper figures (R assets + Inkscape) -> figures/paper/*.pdf|png|svg
	python experiments/run_figure_examples.py
	bash figures/paper/build.sh

# Real-data steps need a manifest (data/manifests/README.md):
#   python experiments/forensics/real_media_null.py  --manifest data/manifests/images.csv --media image
#   python experiments/forensics/heldout_detection.py --manifest data/manifests/images.csv --media image
#   python experiments/forensics/compression_sweep.py --manifest data/manifests/images.csv
#   python experiments/run_real_world_baselines.py --help   (IMC 2025 images, hloc, pycolmap, COLMAP)

clean-cache:
	rm -f experiments/results/forensics/cache_*.csv
