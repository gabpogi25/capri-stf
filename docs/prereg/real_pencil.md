# Pre-registration: pencil test between view subsets on real scenes (IMC 2025)

Status: written and committed **before** any confirmatory run. Date: 2026-10-19. Runner `experiments/run_real_pencil.py`; analysis
`experiments/eval/real_pencil_analysis.py`, which implements the rules below verbatim; tests
`tests/test_real_pencil.py`. All are committed with this file.

## Questions

1. **Size on real scenes (S).** Calibrated on one set of real scenes, does each statistic keep
   its size at nominal 5% on other real scenes? The simulations predict that the raw pencil
   statistic's threshold does not transfer between scenes (`docs/prereg/consistency_test.md`, H1:
   98% under scene shift) and that `acat` does (4.5%).
2. **Combination (C).** Does the Cauchy combination of the pencil statistic and the Sampson error
   keep its size and match or beat the better of the two on every injection family, as it did in
   simulation (`consistency_test`, H3)?
3. **Transfer across scene families (T, descriptive).** Thresholds from the calibration scenes of
   one family, sizes on the test scenes of each family.

## Data and windows

- Scenes: the 23 units of `docs/prereg/real_reconstruction.md` (IMC 2025 training set, at most 75
  images). Tracks come from that benchmark's reconstructions, pipeline `superglue_ransac`, seed
  86,000,000 (`outputs/benchmark_real_batch{1,2}/.../sparse/0`): the control pipeline, not the
  CAPRI one.
- Calibration and test scenes are disjoint: half of the 23 units (11, rounded down) are drawn
  as calibration scenes with seed 87,000,000 (`run_real_pencil.py --split`,
  `experiments/results/real_pencil_split.csv`), the other 12 are test scenes.
- A window: six registered views of one unit and 30 3D points observed in all six. Windows are
  found by a randomized greedy covisibility search (a random first view, then each next view
  drawn among the 3 that keep the most common points); only distinct view sets count; the 30
  points are drawn at random from the common points. Up to 80 windows per unit; a unit with
  fewer qualifying view sets contributes what it has (the count per unit is reported).
- Observations are undistorted with the COLMAP camera and mapped to a 1280-pixel long side (the
  frame of the simulations); views are ordered by image name; subsets A = {0, 2, 4},
  B = {1, 3, 5}.
- Each window is scored unperturbed and once per injection family, at the detectable doses of
  the detector benchmark: jitter 2 px, drift 4 px, warp 2 px, swap 0.05. An injected variant adds
  to the real observations the displacement the family induces on the reprojected COLMAP points
  (`synthetic_data/injection.py`), so the real noise is kept.

## Statistics

`pencil_q10`, `acat` (with the fixed noise scale m0 of the evaluation plan), `sampson_mean`, their
combination (Cauchy combination of the conformal p-values of `pencil_q10` and `sampson_mean`
against the calibration null), and `split_glrt` with sigma^2 estimated on the calibration scenes
(median of sigma_hat_px^2 over their unperturbed windows) and the chi^2_{75} threshold
(3n - 15 = 75 for n = 30). All from `experiments/eval/mv_window.window_statistics`, unchanged.

## Rules (implemented in real_pencil_analysis.py)

- Thresholds: 95th percentile of the unperturbed calibration windows (single statistics);
  conformal p <= 0.05 (combination); chi^2_{75} 95% quantile (GLRT).
- Intention to treat: an abstained window or a NaN value gets the median calibration value of
  that statistic; never a value from the label. Coverage is reported.
- **S:** size on the unperturbed test windows in [3.5%, 6.5%]. Rule R3: power and AUC are
  reported only for statistics with a valid size.
- **C:** supported if the combination's size is valid and, on every injection family, the 95%
  CI of power(combination) - max(power of `pencil_q10`, power of `sampson_mean`) has lower end
  > -0.02, and on at least one family > 0. CI: percentile bootstrap resampling **test scenes**
  (windows of one scene are not independent), 2,000 resamples, seed 87,500,000.
- **T:** for each family with at least 20 calibration windows, its thresholds applied to the
  test windows of every family; sizes reported, no decision.

## Batches

- Batch 1: the units of reconstruction batch 1 (10 units: 5 calibration, 5 test by the draw).
  Interim: no unit, statistic, threshold, analysis or stopping change is made because of it; a
  non-significant interim result is "not yet known". Interim analysis:
  `real_pencil_analysis.py --interim 1 --input .../real_pencil_windows_batch1.csv`.
- Batch 2: the units of reconstruction batch 2 (13 units: 6 calibration, 7 test), run after the
  reconstruction batch 2. The decisions above are made once, on both batches together.
- Window yield in batch 1 (geometry only, no statistic computed; `run_real_pencil.py --batch 1
  --count_only`, confirmatory window seeds): 173 calibration windows from 5 units (baalshamin 80,
  piazza_san_marco 80, kyiv-puppet-theater 5, bike 4, imc2024 dioscuri 4) and 152 test windows
  from 4 of the 5 test units (church 80, another_ET 39, imc2023 dioscuri 30, wall 3,
  stairs_split_2 0). One unit gives over half of the test windows; the scene-level bootstrap is
  why the CIs resample units, not windows.
- Window size and cap. The simulations used 40 points; with 40 points and at most 40 windows per
  unit, batch 1 gave 80 test windows, below the target of about 150-200. Counts of windows alone
  (exploration seeds, no statistic) for 40/30/20 points and caps 40/80 gave 80/111/123 and
  121/153/209 test windows. 30 points with a cap of 80 meets the target while keeping
  3n - 15 = 75 frame-free numbers per window (105 at n = 40, 45 at n = 20). This choice used
  window counts only.

## Seeds (fresh bands, unused in exploration)

Split 87,000,000; windows 87,100,000 + 1000 x (unit index) + window index; injections derived
from the window seed; bootstrap 87,500,000; smoke and profiling runs 87,9xx,xxx (exploration).

## Predictions (stated before the run)

- S: `pencil_q10` likely fails (simulated scene shift: 98%); `acat` and `sampson_mean` may hold;
  the GLRT depends on how well one sigma describes real reprojection noise, likely not.
- C: open.

## What this can and cannot show

Tracks from COLMAP reconstructions are already consistent with one rigid scene up to COLMAP's
own outlier rejection, so the null is "verified real tracks", not raw matches. Injections are
synthetic displacements added to real tracks; real manipulations may differ. One dataset,
scenes of at most 75 images, one matcher. No novelty claim.

## Exploration before registration (motivation, not evidence)

Profiling and smoke runs on reconstructions of the reconstruction smoke run (seed 86,900,000)
and on two batch-1 reconstructions of the earlier, discarded batch-1 attempt (seeds 87,9xx,xxx):
one window takes about 0.5 s with one BLAS thread per process; window yield ranged from 8 to 40
per unit. No statistic value from these runs was used for any choice above.

## Deviations

### 1. Added descriptive analysis: scene-clustered size intervals (after the decisions)

`experiments/eval/real_pencil_size_ci.py` (seed 87,600,000, registered in advance) computes
scene-clustered bootstrap intervals for the sizes. It is not part of the rules above and was
added after the decisions (decision file committed in 37d5343 at 2026-10-19 19:50 UTC; the
script in 68db916 at 21:32 UTC). It is reported as descriptive, enters no decision and changes
no registered number.
