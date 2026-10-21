#!/usr/bin/env bash
# Build the CAPRI graphical abstracts.
#  1. render_assets.R renders the image assets (assets/*.png) from the photo demo results and fills
#     the numbers of the Inkscape layouts (*.template.svg -> *.svg) from the result files.
#  2. Inkscape exports each filled layout to PDF (vector) and PNG (300 dpi).
# Edit the layouts in Inkscape: open *.template.svg (the images are linked, relative paths).
# Run from anywhere: bash figures/abstracts/build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
Rscript figures/abstracts/render_assets.R
cd figures/abstracts
for f in graphical_abstract abstract_mathematics abstract_pipeline; do
  inkscape "$f.svg" --export-type=pdf --export-text-to-path --export-filename="$f.pdf" 2>/dev/null
  inkscape "$f.svg" --export-type=png --export-dpi=300 --export-background=white --export-filename="$f.png" 2>/dev/null
done
