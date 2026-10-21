# Pre-registration: real-image reconstruction benchmark (IMC 2025)

Status: written and committed **before** any confirmatory run. Date: 2026-10-19. Real images: the IMC 2025 training set (Kaggle competition
`image-matching-challenge-2025`, 4.84 GB, downloaded 2026-10-18). Runner
`experiments/run_real_world_baselines.py` (pipelines in `colmap_capri_automator.py`,
`reconstruction/gpu_matching.py`, `reconstruction/match_filtering.py`); scene selection and batch
draw `experiments/draw_real_batches.py`; analysis `experiments/eval/real_reconstruction_analysis.py`,
which implements the decision rule below verbatim. All are committed with this file.

## Question

Does the CAPRI reliability layer, applied after a per-pair RANSAC prefilter, change the accuracy
of COLMAP reconstructions from SuperPoint + SuperGlue matches on real image collections?
Primary comparison: `superglue_ransac_capri` against `superglue_ransac`.

## Units and scene selection

The unit is one IMC scene (`dataset/scene`). An IMC "dataset" can hold several scenes, each with
its own ground-truth frame, plus images labelled `outliers`. Each unit is reconstructed from all
of that scene's labelled images; `outliers` images belong to no scene and are never used.
The training set has 30 units (9-200 images; `run_real_world_baselines.py --list_scenes`).

**Rule:** every unit with at most 75 images is included, with all its images: 23 units.
The 7 units above 75 images are excluded from the confirmatory set; they form the optional
batch 3 below.

**Deviation from the project guidance.** The project's working guidance asks for every scene if the time
budget allows, and otherwise a random draw of at least 10-12 scenes. The author chose instead a
size rule, stated here before any result, because the cost of a unit grows with the square of its
image count (all 30 units: about 39 h on this machine; the 23 units: about 14 h). The rule is
fixed in code (`MAX_IMAGES = 75`) and depends on no result. Scenes are never chosen by hand.

## Batches

The design is run in batches. The batch of each unit is in
`experiments/results/real_reconstruction_batches.csv`, written by `draw_real_batches.py`
(draw seed 86,600,000; `numpy.random.default_rng`).

| Batch | Units | Images | Planning estimate | Role |
|---|---|---|---|---|
| 1 | 10, drawn at random from the 23 | 424 | about 5 h | interim, preliminary |
| 2 | the other 13 | 634 | about 9 h | completes the confirmatory set |
| 3 (optional) | the 7 excluded units | 765 | about 25 h | separate, see below |

Batch 1: ETs/another_ET (10 images), imc2023_haiper/bike (15),
imc2023_heritage/dioscuri (75), imc2023_heritage/wall (43),
imc2023_theather_imc2024_church/church (50), imc2023_theather_imc2024_church/kyiv-puppet-theater (26),
imc2024_dioscuri_baalshamin/baalshamin (49), imc2024_dioscuri_baalshamin/dioscuri (65),
pt_piazzasanmarco_grandplace/piazza_san_marco (68), stairs/stairs_split_2 (23).

Batch 2: ETs/ET (9), fbk_vineyard/vineyard_split_1 (43), fbk_vineyard/vineyard_split_2 (35),
imc2023_haiper/chairs (16), imc2023_haiper/fountain (23), imc2023_heritage/cyprus (30),
pt_brandenburg_british_buckingham/brandenburg_gate (75), .../british_museum (75),
.../buckingham_palace (75), pt_sacrecoeur_trevi_tajmahal/sacre_coeur (75), .../taj_mahal (75),
.../trevi_fountain (75), stairs/stairs_split_1 (28).

Batch 3 (excluded, optional): amy_gardens/peach (200 images, about 11.9 h),
fbk_vineyard/vineyard_split_3 (85, 1.8 h), imc2024_lizard_pond/lizard (90, 2.0 h),
imc2024_lizard_pond/pond (90, 2.0 h), pt_piazzasanmarco_grandplace/grand_place_brussels (100, 2.5 h),
pt_stpeters_stpauls/st_pauls_cathedral (100, 2.5 h), pt_stpeters_stpauls/st_peters_square (100, 2.5 h).

The hour figures are planning estimates from the smoke run (see the script's docstring), not
runtime results.

**Interim rule.** Batch-1 results are interim. No unit, statistic, threshold, analysis or stopping
rule is changed because of them; batch 2 is run whatever batch 1 shows. Interim results are
reported as interim, with their size (10 units, 3 seeds); a non-significant interim result is
"not yet known", not negative. The confirmatory result is the analysis below on all 23 units.

**Batch 3** is not part of the confirmatory set. If run, it uses the same pipelines, seeds and
analysis, and is reported separately as an extension; it does not change the confirmatory result.

The two smoke-run units (ETs/ET, imc2023_haiper/fountain) fell into batch 2 by the draw. Their
smoke results (exploration seed 86,900,000) are not reused; they are rerun with the confirmatory
seeds.

## Pipelines (all with COLMAP incremental mapping and geometric verification, via hloc 1.5 and pycolmap 4.2.1)

| Pipeline | Matching | Filtering |
|---|---|---|
| `sift` | SIFT (hloc `sift`), NN-ratio, exhaustive pairs | none |
| `superglue` | SuperPoint (`superpoint_max`, up to 4,096 keypoints) + SuperGlue (outdoor weights), exhaustive pairs | none |
| `superglue_ransac` | as `superglue` | per-pair USAC_MAGSAC fundamental matrix, 3 px, confidence 0.99 |
| `superglue_ransac_capri` | as `superglue` | as `superglue_ransac`, then the CAPRI layer: `conditioning_threshold` 1000, `max_tuples` 150, `empty_pair_policy` `keep_ransac` |

- Features and matches are computed once per unit and shared by the three SuperGlue pipelines and
  all seeds (likewise for SIFT), so the SuperGlue pipelines differ only in filtering.
- Matching runs on the GPU in full precision (float32; TF32 matrix multiplication off, the
  PyTorch default). It gives the same matches as hloc's own matcher on the same features.
- Filtering runs in parallel worker processes; each pair reseeds, so the output does not depend
  on the number of workers.
- The committed default settings are used; no setting is tuned on any unit.

## Seeds (fresh bands, unused in exploration)

- Reconstruction: 86,000,000 + k, k = 0, 1, 2 (three seeds per unit and pipeline). The seed drives
  the RANSAC sampler (`cv2.setRNGSeed`, per pair) and the CAPRI tuple sampling. `sift` and
  `superglue` do not use it; their seed rows measure COLMAP run-to-run variation only.
- Bootstrap: 86,500,000. Batch draw: 86,600,000. Smoke and timing runs: 86,900,000 (exploration).
- The band 86.0e6-86.9e6 was unused on 2026-10-19 (code, docs and `docs/prereg/` checked).

## Metrics (per unit, pipeline and seed; one row each)

- `mAA_at_10` (primary): mean over thresholds 1..10 degrees of the fraction of image pairs whose
  pose error, max(rotation error, translation-direction error), is below the threshold.
  **Every ground-truth pair of the unit counts; a pair with an unregistered image is a miss**
  (intention to treat, rule R4). Only the largest COLMAP model is scored.
- Secondary: registered images, median pose error, accuracy at 5/10/20 degrees, camera-centre RMSE
  after Sim(3) alignment (ground-truth units, not metres), matches in / after RANSAC / out, pairs
  where CAPRI accepted no tuple. A run with no model is a row with zero registered images and the
  `Failure` column set; it is never dropped.

## Primary comparison and decision rule (on all 23 units)

Paired difference per unit: D = mean over the three seeds of mAA_at_10(`superglue_ransac_capri`)
minus the same for `superglue_ransac`. Estimate: mean of D over the 23 units, with a 95% percentile
bootstrap CI over units (10,000 resamples, seed 86,500,000).

- **Benefit supported** if the CI lower end > 0.
- **Harm** if the CI upper end < 0.
- **No difference shown** otherwise; "equivalent within 0.01 mAA" if the CI lies inside
  [-0.01, 0.01], else "inconclusive".

Descriptive only: the same D for the secondary metrics; Wilcoxon signed-rank p-value on D; all
four pipelines side by side; failures per pipeline. Interim analysis after batch 1:
`real_reconstruction_analysis.py --interim 1 --input .../real_world_baselines_batch1.csv`; it
computes the same numbers and writes no decision.

## Runtime

The per-stage times in the result rows are not the runtime result: downloads for the forensic
study (Track 2) run alongside. Runtime is measured separately on an idle machine, in its own run.
Machine: Windows 11 Home 10.0.26200, Intel i5-12450H, 23.8 GB RAM, NVIDIA RTX 3050 6 GB Laptop GPU
(driver 616.92), PyTorch 2.11.0+cu128 (CUDA 12.8), Python 3.12.10; COLMAP 3.9.1 (CUDA) is installed,
but the sparse stage uses pycolmap's CPU mapper. Every row records platform, GPU and PyTorch.

## Prediction (stated before the run)

No difference. In the smoke runs (exploration) CAPRI removed 6-608 of the RANSAC inliers per scene
(under 2%), so the layer changes little in what COLMAP sees, and COLMAP's own verification follows.

## What this can and cannot show

One dataset (IMC 2025 training set), scenes of at most 75 images, one matcher, default settings,
one machine. A null result does not show that CAPRI's diagnostics are useless elsewhere; a positive
result holds for these scenes and settings only. No novelty claim.

## Exploration before registration (motivation, not evidence)

Smoke runs on ETs/ET and imc2023_haiper/fountain, seed 86,900,000
(`experiments/results/smoke_real_world_baselines.csv`, `smoke_real_world_baselines_gpu.csv`). They
found faults in the runner, fixed before this registration: ground truth mixed across the scenes of
a dataset, the wrong COLMAP model scored, unregistered images left out of the pose metrics, and
hloc's data-loading workers hanging on Windows.

## Deviations

Any change after the registration commit (7d87d82) is recorded here, with its date, before the
run it affects.

### 1. OpenCV error on a degenerate pair (2026-10-19, batch 1 stopped; no result file written)

Batch 1 stopped after 30 of 120 runs, in the first `superglue_ransac` run of
imc2023_heritage/dioscuri: `cv2.findFundamentalMat` (USAC_MAGSAC) raised an internal assertion
(`!model.empty()`) on one image pair instead of returning no model. The filtering code now treats
that error exactly like "no fundamental matrix", which it already handled (the pair keeps no
matches), and counts such pairs in a new column `Pairs_RANSAC_Error`. On the matches of that
scene, seed 86,000,000, `ransac` mode, 1 of 2,775 pairs is affected. No pipeline, setting,
metric, unit or analysis changes.

The 30 finished rows were written by the registered commit, and the runner refuses to mix rows
from two code versions, so batch 1 is rerun from the start under the fixed commit, with the same
seeds. The features and matches already computed for batch-1 units are reused: they were made
by matching code this fix does not touch, are independent of the seed, and the runner reuses
them by design (a completed-matching marker per unit). The metrics of the 30 rows were not
looked at (only which unit, pipeline and seed had finished); they are discarded with the
progress file. Two of their sparse models were read by an exploration smoke run of the
pencil-test runner (window search only), which uses no reconstruction metric.

### 2. Analysis check on a descriptive metric (2026-10-19, interim analysis of batch 1)

The interim analysis stopped before writing any output: for `Median_Pose_Error_deg` (descriptive),
a unit whose median pose error is infinite in both pipelines (most pairs unregistered) gives
inf - inf = NaN, and a check meant to detect a missing pipeline treated that NaN as missing. The
check now tests that both pipelines have runs for every unit; an undefined difference is dropped
for that descriptive metric, whose row reports the number of units used (`N_Units`). The primary
metric `mAA_at_10` is always finite, and its rule is unchanged. No number was seen before the
fix.
