# Pre-registration: consistency test, version 2 (simulation)

Status: written and committed **before** any confirmatory run of the experiment below.
Date: 2026-10-18. Simulated data only.
Runner `experiments/eval/consistency_test_windows.py`;
the decision rules are implemented verbatim in `experiments/eval/consistency_test_analysis.py`,
committed with this file.

## Questions

1. **Geometry transfer of a standardised statistic (H1, H1b).** The mathematics
   (Proposition 4.4) shows that standardised discrepancies are marginally pivotal to first
   order under isotropic noise in the chart where the gradients are taken. Real reconstructions
   do not have isotropic noise in any known chart. H1 asks whether standardising in a common
   canonical frame (Tyler) is enough for the null to transfer from scene S0 to scene S1. H1b
   replicates, on fresh seeds, the transfer seen for `acat` in the null calibration of
   `docs/prereg/evaluation_plan.md`.
2. **Noise misspecification (H2).** Real pipelines must estimate the noise level. With a
   plug-in sigma_hat = f * sigma, statistics that scale with sigma shift their threshold by f,
   those that scale with sigma^2 by f^2. H2 asks whether the pencil statistic's size is less
   sensitive to a misspecified noise level than the split GLRT's.
3. **Combination (H3).** In the detector benchmark the Sampson error was stronger on jitter and
   drift and the pencil statistic on warps and swaps. H3 asks whether a Cauchy combination of
   their conformal p-values keeps its size and is at least as powerful as the better of the two
   on every injection type, and more powerful on at least one.
4. **Descriptive.** AUCs of all statistics per injection type; sizes at 0.25 and 1 px with the
   correct sigma; the frame-coordinate statistic on fresh seeds (exploration found it much
   weaker than the pencil family; `experiments/results/frame_statistic_exploration_summary.csv`).

## Setup

Windows as in `docs/prereg/evaluation_plan.md` (`experiments/eval/mv_window.py`): n = 40 points,
six calibrated views, interleaved split A = {0, 2, 4}, B = {1, 3, 5}, projective reconstruction of
each subset, joint bundle adjustment for the GLRT. Scene S0: points uniform in [-1, 1]^3, cameras
on a 30-degree arc at radius 6. Scene S1: 60 Gaussian points (sd 0.5), 20-degree arc (as the
shifted scene of the null calibration).

| Cell | Scene | Family, dose | sigma (px) | Windows |
|---|---|---|---|---|
| cal | S0 | rigid | 0.5 | 400 |
| test | S0 | rigid | 0.5 | 400 |
| sig025 / sig1 | S0 | rigid | 0.25 / 1.0 | 200 / 200 |
| shift | S1 | rigid | 0.5 | 200 |
| jitter / drift / warp / swap | S0 | 2 px / 4 px / 2 px / 0.05 | 0.5 | 400 each |

The doses are the detectable doses of the detector benchmark.

Statistics: `pencil_q10`; `std_native_q10` and `std_tyler_q10` (Q_0.1 of
D_t / (|g_t(A)| + |g_t(B)|) on 20,000 sampled pencils, in native frames and after Tyler
whitening); `acat` (as in mv_window); `sampson_mean`; `split_glrt` (divided by the generating
sigma^2); `frame_q50` (median of the 3(n - 5) frame-coordinate discrepancies).

## Rules (implemented in consistency_test_analysis.py)

- Thresholds: 95th percentile of the 400 cal windows per statistic.
- Intention to treat: abstained windows and NaN values get the median cal value; never imputed
  from the label. Coverage is reported.
- Size band: [3.5%, 6.5%] at nominal 5%.
- **H1:** size of `std_tyler_q10` on shift in the band. **H1b:** the same for `acat`.
  Descriptive: `split_glrt` on shift against the chi^2 quantile with 3n - 15 = 165 degrees of
  freedom.
- **H2:** sigma_hat = f sigma, f in {0.5, 0.75, 1.5, 2}. Threshold c set at f = 1 on cal; at f
  the test rejects above c f (pencil_q10, std_tyler_q10) or c f^2 (split_glrt, sampson_mean);
  split_glrt also against chi^2_{105} times f^2. Size on test. **Supported** if at f = 0.75 and
  at f = 1.5, |size - 0.05| of `pencil_q10` is smaller than that of `split_glrt` (empirical
  threshold).
- **H3:** conformal p-values of `pencil_q10` and `sampson_mean` against cal; Cauchy combination;
  reject at combined p <= 0.05. **Required:** size on test in the band. **Supported** if on every
  injection type the paired-bootstrap 95% CI (2,000 resamples, seed 81,900,000) of
  power(combined) - max(power of the two single statistics at their own thresholds) has lower end
  > -0.02, and on at least one type > 0.
- Descriptive AUCs with paired bootstrap CIs against `pencil_q10` (same seed).

## Predictions (stated before the run)

- H1: likely **not** supported. A smoke run (8 windows per cell, exploration seeds
  141,000,000 + k; `experiments/results/smoke_consistency_test_windows.csv`) put the median of
  `std_tyler_q10` on S1 about 1.5 times its cal median, consistent with the noise in the Tyler
  frame depending on the camera geometry.
- H1b: expected supported (replication).
- H2: expected supported by the first-order scaling (f against f^2), but both statistics may
  leave the band at f = 0.75 and 1.5; H2 compares distances to 0.05, not band membership.
- H3: open.

## What this can and cannot show

Simulation only; one camera model; pixel noise. No real-image claim follows. H2 uses an
externally misspecified noise level, not one estimated from the same windows (that case is the
F-ratio, already shown to have no power). Thresholds are calibrated on S0; transfer is tested
to one other scene only.

## Seeds (fresh; registry addition)

Windows: 81,000,000 + k, k = 0..3399 in the task order of the runner. Bootstrap: 81,900,000.
Smoke runs: 141,000,000 + k (exploration). The band 81.0e6-84.0e6 was unused on 2026-10-18.

## Exploration before registration (motivation, not evidence)

`experiments/math_lab/explore_frame_statistic.py` (seeds 90.2e6) and the smoke run above. The
smoke run also showed `acat` almost unchanged on S1 and the n = 40 empirical GLRT threshold not
transferring to n = 60 (degrees of freedom change), which is why the chi^2 comparison with
165 degrees of freedom is reported.

## Deviations

### Seed range miscounted in this document (2026-10-18, run started, no output written yet)

The Seeds section says windows use k = 0..3399. The cell table above sums to 3,000 windows
(400 + 400 + 200 + 200 + 200 + 4 x 400), and the runner, committed with this file, uses
k = 0..2999 in the task order of the table. The table and the runner define the grid; the range
81,000,000-81,002,999 is used. No result file existed when this was noticed.
