# Pre-registration: forensic video study, real-media null and held-out detection (GenVideo)

Status: written and committed, with `configs/default.yaml` frozen at the same commit,
**before** any confirmatory forensic run. Date: 2026-10-19.
Runners `experiments/forensics/real_media_null.py`, `heldout_detection.py`
(unchanged decision code; gates from `configs/default.yaml`); statistics
`experiments/forensics/media_statistics.py`; batch draw `scripts/draw_forensic_batches.py`.

## Questions

1. **Real-media null (gate A).** With empirical nulls fitted on authentic clips (`train_null`),
   does the video statistic reject at most the nominal 5% of other authentic clips (`val`):
   rejection rate in [3.5%, 6.5%]?
2. **Held-out detection (gate B).** On generators never used for any fit or threshold (Gen2,
   MoonValley, Sora), against authentic test clips: AUC >= 0.70 with the cluster-bootstrap CI
   above 0.5 for at least one held-out generator, **and** a metadata-only classifier at
   AUC <= 0.55 (otherwise any detection could be a format shortcut).

## Data

- Generated: GenVideo-Val (ModelScope `cccnju/Gen-Video`, `GenVideo-Val.zip`, sha256 verified),
  10 generators, 8,302 MP4/MOV clips (282 GIF clips are not read by `build_manifest.py`).
- Authentic: the first 4,000 clips of GenVideo-100K's real archive (`cccnju/GenVideo-100K`,
  `Real_part_aa`, Youku clips, streamed by `scripts/stream_genvideo_real.py`). The published
  sha256 covers the whole 30 GB part and cannot be checked on a partial stream; every written
  clip decodes and has a sha256 in the manifest. GenVideo-Val's own real side (MSR-VTT previews,
  298x224 at 3 fps) is **not** used: it differs from the generated side in resolution, frame rate
  and length so strongly that no fair test is possible on it.
- Manifest `data/manifests/videos_v2.csv` (`build_manifest.py --held-out-generators sora gen2
  moonvalley`, salt `capri-forensics-v1`): authentic sources hashed into train_null 1,582,
  val 809, cal 802, test 807; the 7 other generators -> cal, the 3 held-out generators -> test.
  Licences: Youku/GenVideo real clips and GenVideo-Val are not redistributed; DeMamba states no
  dataset licence (see `scripts/fetch_forensic_data.py`).

## Known format differences (stated before the run)

Authentic clips are 640x360 (some 360x640, 640x640), 24 fps, 10-16 s; generated clips are
512-768 px high, 8-50 fps, 2-4 s. Pixels are never resampled before geometric analysis
(`ingest.geometry_resize: none`). Harmonisation: only the first 4 s of every clip are analysed
(`max_frames_per_clip` 60 at 15 fps). Resolution and frame rate stay different, and the
metadata audit reads the file's own metadata, so **gate B's metadata condition is expected to
fail**; that outcome is reported as a fail of the gate, not explained away.

## Configuration (frozen in `configs/default.yaml` at the registration commit)

Changed on 2026-10-19 from the earlier defaults, on authentic `val` and generated `cal` clips
only (exploration runs below), never on `test`:
- `ingest.video_sample_fps` 5 -> 15 and `ingest.min_shot_len_frames` 24 -> 16: a six-view
  window (spacing 3) then spans about 1 s; at 5 fps it spanned 3.2 s, longer than most
  generated clips, so no window could form.
- `ingest.max_frames_per_clip` 150 -> 60: the harmonisation above.
- `global.base_seed` 42000 -> 88,000,000 (fresh band).
All gates unchanged: `real_media_null_band` [0.035, 0.065] at nominal 0.05,
`heldout_video_min_auc` 0.70, `heldout_max_metadata_auc` 0.55. The earlier simulated fixture
results keep the config hash they were made with.

## Statistic and coverage

Per clip: pencil drift statistics of every usable six-view window (rigid tracks, parallax gate)
and frame statistics (`media_statistics.video_table`). The video score is the block-Fisher
combination of window and frame p-values against the gate-A nulls. A clip with no usable window
and no usable frame abstains; abstained clips are excluded from AUC and **coverage is reported
per class and per generator** (rule R4; never imputed from the label). In exploration the pencil
branch formed windows in 8 of 20 authentic and 4 of 20 generated clips, so most of the score will
come from frame statistics; window coverage is reported separately so it is visible how much of
the result is the pencil statistic.

## Batches

Drawn by `scripts/draw_forensic_batches.py` (seed 88,600,000) from `videos_v2.csv`;
summary `experiments/results/forensics_video_batches.csv`.
- Batch 1 (1,556 clips): 300 authentic each from train_null, val and test; 300 generated from
  cal (30-76 per generator, all 7); test generators Gen2 150, MoonValley 150, Sora 56 (all).
- Batch 2: every other clip of the manifest (10,746).
Batch 1 is interim: no clip, statistic, threshold, gate, analysis or stopping change is made
because of it, and batch 2 is run whatever batch 1 shows. Interim results are reported as
interim with their size; a non-significant interim result is "not yet known". The gates are
decided once, on batches 1 and 2 together.

## Seeds

Statistics: `global.base_seed` 88,000,000 (+ clip index within the manifest that is run).
Runner seeds: `real_media_null.py --seed 88,100,001`, `heldout_detection.py --seed 88,100,002`.
Batch draw 88,600,000. Exploration 88,9xx,xxx. Band 88.0e6-88.9e6 registered in advance.

## Run order

1. `real_media_null.py --manifest data/manifests/videos_v2_batch1.csv --media video --seed 88100001`
2. `heldout_detection.py --manifest data/manifests/videos_v2_batch1.csv --media video --seed 88100002`
3. Batch 2 later, then the decision on the full set. `compression_sweep.py` re-encodes still
   images; it belongs to the image branch (Synthbuster with RAISE as the authentic side) and is
   not part of this video study.

## Predictions (stated before the run)

- Gate A: open. The frame statistic's null was calibrated on simulated scenes only.
- Gate B: the metadata condition fails (format differences above), so gate B fails as a whole
  whatever the AUCs; AUCs per held-out generator are reported with that qualifier.

## What this can and cannot show

One authentic source (Youku clips from GenVideo-100K's first archive part), one generated
benchmark, clips of 2-16 s, many with little camera motion. A pass of gate A shows a calibrated
null on these clips only. No result here supports a claim about real-world detection of
generated video. No novelty claim beyond "to our knowledge" with prior work credited.

## Exploration before registration (motivation, not evidence)

`experiments/forensics/smoke_video.py` (seeds 88,900,000 + ...): three runs of 20 or 40 clips,
looking at time per clip, number of windows and frames, and abstention only; no statistic value,
p-value or AUC was looked at. **The first two runs (20 clips each) drew from all splits, including
test-split clips of held-out generators (Gen2, MoonValley) and the GenVideo-Val MSR-VTT reals;** the third and later
exploration used authentic `val` and generated `cal` clips only. They found an OpenCV USAC
crash on degenerate tracks (fixed, f2e9a2a) and the window-span problem above.

## Deviations

None yet.
