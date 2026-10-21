#!/usr/bin/env bash
# Build the illustrative paper figures (teaser, pencil, design, pipeline).
#  1. experiments/run_figure_examples.py must have written experiments/results/figure_*.csv, and the
#     graphical abstracts must be built (bash figures/abstracts/build.sh): some overlays are shared.
#  2. render_assets.R renders assets/*.png and fills *.template.svg -> *.svg from the result files.
#  3. The pipeline figure is the filled pipeline abstract without its title row (image links rebased).
#  4. Inkscape exports each layout to PDF (vector, text as paths) and PNG (300 dpi).
# Edit the layouts in Inkscape: open *.template.svg (images are linked, relative paths).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
Rscript figures/paper/render_assets.R
python3 - <<'PY'
import re
s = open("figures/abstracts/abstract_pipeline.svg").read()
s = re.sub(r'\n\s*<text x="30" y="5\d" class="title">.*?</text>', "", s, count=1, flags=re.S)
w, h = re.search(r'viewBox="0 0 (\d+) (\d+)"', s).groups()
s = s.replace(f'viewBox="0 0 {w} {h}"', f'viewBox="0 68 {w} {int(h) - 68}"').replace(f'height="{h}"', f'height="{int(h) - 68}"', 1)
s = s.replace('xlink:href="assets/', 'xlink:href="../abstracts/assets/').replace('xlink:href="photos/', 'xlink:href="../abstracts/photos/')
open("figures/paper/pipeline.svg", "w").write(s)
PY
cd figures/paper
for f in teaser pencil design pipeline; do
  inkscape "$f.svg" --export-type=pdf --export-text-to-path --export-filename="$f.pdf" 2>/dev/null
  inkscape "$f.svg" --export-type=png --export-dpi=300 --export-background=white --export-filename="$f.png" 2>/dev/null
done
