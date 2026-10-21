# Pre-registration: forensic video study on FaceForensics++ (format-matched real and manipulated)

Status: written and committed **before** any confirmatory run. Date: 2026-10-20. Runners
`experiments/forensics/real_media_null.py` and `heldout_detection.py`, unchanged; configuration
`configs/default.yaml` exactly as frozen by `docs/prereg/forensics_video.md` (config hash
09e31f4ff8d4c02d); manifest `scripts/build_ffpp_manifest.py`.

## Why this study

In the GenVideo study, authentic and generated clips differ in resolution, frame rate and length;
a metadata-only classifier separated them (AUC 0.993, batch 1), so detection AUCs there cannot be
read as evidence about content. FaceForensics++ manipulated videos are made from the same
original videos and encoded the same way (all 1280x720 at 25 fps in the downloaded set), so the
metadata condition of gate B can be met and a detection AUC can mean something.

## Questions

1. **Gate A (real-media null)** on FF++ originals: rejection rate on `val` originals, with the
   null fitted on `train_null` originals, in [3.5%, 6.5%] at nominal 5%.
2. **Gate B (held-out detection)**: for each of Deepfakes, Face2Face, FaceSwap, NeuralTextures
   against the authentic test originals, AUC >= 0.70 with the cluster-bootstrap CI above 0.5 for
   at least one method, **and** metadata-only AUC <= 0.55.

## Data

- FaceForensics++ (Rossler et al. 2019), downloaded by the author with FaceForensics' own script
  after access approval and acceptance of its terms of use: server EU2, compression c23, the first
  200 videos of `original_sequences/youtube` and of each of the four manipulation methods in the
  script's file order (`-n 200`), 1,000 videos in all (1.8 GB). All decode. Not redistributed.
- The first 200 originals form 100 FF++ pairs (A, B); every manipulated video A_B uses target A
  and the face of B, both among the 200 originals (checked).
- Manifest `data/manifests/ffpp_c23.csv` (`build_ffpp_manifest.py`, salt `capri-forensics-ffpp-v1`):
  the split unit is the **pair** (both originals and all eight manipulated videos of a pair in
  one split), fractions 0.4 / 0.2 / 0.2 / 0.2: train_null 40 pairs (80 originals), val 19 (38),
  cal 21 (42), test 20 (40 originals, 40 manipulated videos per method). All four methods are
  held out (none is used for any fit or threshold); the gates read only authentic items outside
  test (gate A) and the test split plus authentic val items (gate B).

## Statistic and coverage

As in `docs/prereg/forensics_video.md`, with the same frozen configuration: per clip, pencil
drift statistics of usable six-view windows and frame statistics; score = block-Fisher
combination against the gate-A nulls; abstained clips excluded from AUC, coverage reported per
class and method. In exploration (val and cal clips only) the pencil branch formed a window in
1 of 20 FF++ clips (talking heads, little camera motion), so **this study tests the frame
statistic, not the pencil statistic**; window coverage is reported so this is visible.

## Batches

- Batch 1: all 1,000 downloaded videos. Interim: no clip, statistic, threshold, gate, analysis
  or stopping change is made because of it.
- Batch 2: the next 200 originals and 800 manipulated videos in FF++'s own file order
  (`download.py -n 400`, keeping the first 200), placed by the same manifest rule. The file order
  is fixed by the dataset, not chosen by us. The gates are decided on batches 1 and 2 together.
  If batch 2 cannot be downloaded, batch 1 is reported as preliminary, not as the decision.

## Seeds

The frozen configuration's `global.base_seed` (88,000,000) seeds the per-clip statistics (+ clip
index within the FF++ manifest), as in the GenVideo study; it is not changed, because the
configuration is frozen. The data differ, and the seed only drives the random sampling inside
the statistic (pencils, RANSAC). Runner seeds: `real_media_null.py --seed 88200001`,
`heldout_detection.py --seed 88200002` (band 88.2e6, registered in advance).

## Predictions (stated before the run)

- Gate B's metadata condition: met (AUC near 0.5): same encoding for both classes.
- Gate A and the detection AUCs: open. The frame statistic looks for violations of perspective
  geometry in straight lines and vanishing points; face manipulations change a small face region
  and leave the background, so low detection power is plausible.

## What this can and cannot show

Face manipulations only (four methods from 2016-2019), c23 compression, 40 test originals and
40 manipulated videos per method in batch 1 (wide CIs). A pass would show that the frame
statistic separates these face manipulations at this compression; it says nothing about modern
generators. No novelty claim beyond "to our knowledge" with prior work credited.

## Exploration before registration (motivation, not evidence)

`experiments/forensics/smoke_video.py` on 10 authentic and 10 manipulated FF++ clips from the val
and cal splits only (seed 88,900,000): time per clip about 10 s, windows and frames counted, no
statistic value looked at. No test-split clip was used.

## Deviations

### 1. Empty pencil branch: frame statistics only (2026-10-20, before any gate result)

The first gate-A run stopped before writing any output: the 80 `train_null` originals gave 2
usable pencil windows, and an empirical null needs at least 20 authentic values. The window null
therefore cannot be fitted and the pencil branch is empty. As this registration states, the study
tests the frame statistic. So, when fewer than 20 authentic windows exist, gate A computes the
rejection rate on the frame statistics of the `val` originals against the frame null fitted on
`train_null` (`gate_a_statistic: frame` in the output), and the block-Fisher score of gate B uses
the frame p-values only. `real_media_null.py` and `heldout_detection.py` were changed for exactly
this case (no window null); with at least 20 windows, as in the GenVideo study, they behave as
before. Nothing else changes: the frozen configuration (hash 09e31f4ff8d4c02d), splits, thresholds
and seeds 88,200,001 (gate A) and 88,200,002 (gate B). Outputs carry the tag `_ffpp`. No statistic
value, p-value or AUC had been seen.
