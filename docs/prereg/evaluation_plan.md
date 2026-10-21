# Pre-registration: synthetic evaluation

Status: written and committed **before** any confirmatory run of the experiments below.
Date: 2026-10-12. All data are synthetic.

This file builds on an earlier evaluation shortlist and adds the arms chosen on 2026-10-12 (noise-model check, conformal p-values, ACAT, asinh, e-values, additive gauge fit,
GLRT, D-optimal selection, A-robustness arms). Every rule below is implemented in the named script;
the analysis scripts contain the decision rules verbatim and are committed with this file.

## Rebase notes (what changes against the shortlist)

- The shortlist's validity audit targets (`run_pilot_gates.py`, `run_exp14_gate_c.py`,
  `run_statistical_eval_*`, `video/video_pipeline.py`, `synthetic_video_generator.py`, bracket
  blends, the vp stub) do not exist on this branch or were already fixed. validity audit becomes a short
  re-audit of the result files that exist here (score audit).
- The shortlist's "exp14 / the held-out video gate" is not this branch's matched degeneracy experiment (degeneracy test, the coplanarity component).
- Primary statistic: the pencil drift statistic `pencil_q10` (Q_0.1 of |d log|CR|| between the
  two subset reconstructions; docs/MATHEMATICS.md Section 4). Delta_J is the legacy comparison
  (`deltaj_med`). rho cannot be computed between frame-free reconstructions (it needs a common
  chart; MATHEMATICS.md Section 2), so the shortlist's "rho flat" control is replaced by the global_projective
  blind-spot control and the noise-free global_projective check.
- The F6 "bracket blend" family is dropped: the blend generator was removed from this branch as
  non-physical. No statistic here is evaluated on blends.
- scipy, scikit-learn and OpenCV are installed in the container; the projective BA is a dense LM
  with an analytic Jacobian (`forensics/video/fast_ba.py`), because scipy's finite-difference
  solver took 2-15 s per window and stopped early on inconsistent windows (exploration).
- Rules kept from the shortlist: size before power, abstentions never imputed from the label
  (ITT with the median calibration score; coverage reported), thresholds frozen on calibration
  data, paired comparisons (DeLong / paired bootstrap), Bonferroni where several claims are made.
- Not run here (need real data, user's machine): real-null go/no-go, shortcut and leakage controls, real-data detection, selective prediction, single-image statistics real arm, degradation sweep, calibrated fusion.
  validity audit check 6 (construction swap) is not run; noise-model check plays that role for the drift test only.

## Common setup

Scene: n = 40 points uniform in [-1, 1]^3; six calibrated cameras (f = 800 px, 1280 x 720) on a
30-degree arc at radius 6 (`synthetic_data/injection.py`). Window = 6 views; subsets A = {0,2,4},
B = {1,3,5} (interleaved). Each subset: USAC-MAGSAC F, canonical cameras, DLT, resection, LM
projective BA; joint six-view BA from subset A. Statistics per window
(`experiments/eval/mv_window.py`), larger = more inconsistent:

| Statistic | Definition |
|---|---|
| pencil_q10 (primary), pencil_q50, pencil_q90 | quantiles of |d log|CR|| over 50,000 sampled pencils |
| asinh_q10 | Q_0.1 of |asinh CR(A) - asinh CR(B)|, signed CR, 20,000 pencils |
| acat | -log10 ACAT p of z_t = D_t / (m0 (g_t(A) + g_t(B))), two-sided normal p; m0 = 2.198e-4 (fixed from exploration, Section 6) |
| gauge_q90, gauge_med | additive gauge fit d_q = c + sum_{k in q} s_k (Huber IRLS) on 20,000 sampled log-brackets; Q_0.9 / median of |residual| |
| deltaj_med | legacy: median standardized max|Delta_J| over 100 gated six-tuples |
| registration_rms | RMS sine residual of a DLT projective registration in Tyler frames |
| split_glrt | (G_all - G_A - G_B) / sigma^2, known sigma (= "independent sigma", the shortlist's independent-sigma rule) |
| glrt_all | G_all / sigma^2 |
| f_ratio | sigma-free F-ratio with df1 = 3n - 15 = 105, df2 = 2(3n - 18) = 204 |
| sampson_mean | mean squared Sampson error (px^2) of MAGSAC F over consecutive view pairs |
| magsac_outlier | 1 - mean MAGSAC inlier ratio (1 px) over consecutive pairs |

Abstention: the view-0/1 homography (MAGSAC, 2 px) has >= 90% inliers (planar), or reconstruction
fails. A numerical failure inside one statistic group sets that group to NaN and is logged in
`errors`; it never drops the window.

Families (`synthetic_data/injection.py`): rigid; global_projective, an exact projective map H(W) in subset B's
views (provable blind spot); jitter, per-frame jitter; drift, coherent linear drift; warp, a local smooth warp
varying as sin(pi v / 5); jitter/drift/warp move 30% of the points; swap, nearest-neighbour track swaps in views
3-5 (dose = fraction of tracks); planar, a rigid plane (must abstain). Dose of global_projective/jitter/drift/warp = RMS induced pixel
displacement over moved tracks and all views (bisection, 1e-3 px).

## Experiments, sizes, seeds and decision rules

### Score audit (validity audit checks 1-5) - `score_audit.py`
Files: exp6b, exp14 (test), exp11b (test), exp8b (test), true_synthetic_baselines, exp5, exp9, f6.
Checks: (1) AST scan of experiment/algorithm/forensics sources for a `*score*` function that reads
a label name; hits reviewed by hand are listed in the script (`REVIEWED_NOT_SCORERS`), an
unreviewed hit is a label path; (2) floor fraction P(|s| <= 1e3 eps) per class (non-negative
unit-scale scores only); (3) U = #unique/#items; (4) permutation test of |AUC - 0.5|, B = 1000,
Phipson-Smyth p; (5) abstention leak |AUC(non-finite, label) - 0.5|.
Verdict: RETAINED iff no label path, F_floor <= 0.5 in at least one class, U >= 0.05, p < 0.01,
leak <= 0.10; RETRACTED if check 1 or 2 fails; else INCONCLUSIVE.
**Self-test (must fire, else the audit is broken and nothing else is reported):** rho of ONE point
set flagged by check 2; a planted label-reading scorer flagged by check 1; a planted abstention =
label flagged by check 5. Seed 42021.
Prediction (written before running): every listed score RETAINED except exp5 rho (independent-
estimation model, AUC well above 0.5 in Table 5) which is RETAINED too; f6 LS/Rho INCONCLUSIVE
(F6 AUC about 0.5).

### Noise-model check - `noise_model.py`
2000 paired scenes (seeds 31,000,000 + i), 12 points, measurements triangulated from 0.5 px pixel
noise (views {0,2,4} and {1,3,5}, true cameras) vs isotropic 3D noise with the same per-scene RMS;
label 1 = B displaced uniformly in a ball of radius f rho, f ~ U(1.5, 5). Methods: pencil_q10,
registration_rms, median_tuple_dJ (scores / rho). Paired bootstrap 2000 (seed 40,000,000).
**Material change** iff for pencil_q10 |AUC_tri - AUC_iso| > 0.02 or |TPR5 diff| > 0.05 with the
95% CI excluding 0, or the pencil-minus-registration AUC difference has CIs of opposite sign in
the two arms. If material, every pencil drift experiment number carries the caveat "isotropic chart noise model".

### Injection grid and kill switch - `injection_grid.py`, analysis `detector_benchmark.py`
Grid at sigma = 0.5: rigid calibration 400, rigid test 400; global_projective/jitter/drift/warp x doses {0.5, 1, 2, 4, 8, 16} px x 100;
swap x {0.05, 0.1, 0.2, 0.3} x 100; planar x 100. sigma = 0.25 and 1.0: rigid calibration 200, test 200;
jitter/drift/warp at 8 px x 100. Seeds 32,000,000 + k (grid order), checks 32,900,000 + i.
Thresholds: 95th percentile of covered rigid calibration windows per sigma and statistic.
**Kill switch (continuation)**, for pencil_q10: one-sided Jonckheere-Terpstra p < 1e-3 on >= 2 of
jitter, drift, warp AND power >= 0.8 at 8 px, sigma = 0.5, on >= 1 of them. If it fails: stop the pencil
statistic's forensic line and report.
**Full pass** (pencil_q10): for each of jitter/drift/warp/swap, JT p < 1e-3, Spearman(dose, median statistic) >= 0.9
and power >= 0.8 (8 px at sigma = 1.0 for jitter/drift/warp; swap fraction 0.3 at sigma = 0.5 for swap); planar
abstains >= 90%; rigid test size in [3.5%, 6.5%] at every sigma.
**Controls:** global_projective power <= 0.10 at every dose, reported for every statistic; noise-free rigid and global_projective
windows must give pencil_q10 < 1e-6 (20 each; if global_projective fails this, the generator is wrong). Jacobian
rank: d(log|CR| family, n = 12)/d(72 image coordinates of one subset) has rank 3n - 15 = 21 on
>= 95% of 200 configurations, decided by s_21 stable (< 1% change) between h = 1e-6 and 1e-7 and
s_22 falling >= 30x (finite-difference error). A fixed s_21/s_22 >= 1e6 is not attainable
(s_21/s_1 ~ 3e-4, exploration).
Prediction: kill switch passes; global_projective "power" may exceed 0.10 for some statistics at large doses
because H changes the triangulation conditioning (exploration: 0.13-0.27 on 15 windows); the
noise-free check separates this from a bug.

### Null calibration - `null_calibration.py`, analysis `null_analysis.py`
rigid windows at sigma in {0.25, 0.5, 1, 2} px, 600 each (indices < 300 calibration half), plus 200
shifted-scene windows (60 Gaussian points, 20-degree arc, sigma = 0.5). Seeds 33,000,000 + k.
(a) Analytic tests: f_ratio (F(105, 204)), split_glrt (chi^2_105), glrt_all (chi^2_309): size at
alpha 0.01/0.05/0.10 with Clopper-Pearson CIs, KS D; pass iff size(0.05) in [3.5%, 6.5%] at every
sigma. Misspecified control: split_glrt at 0.5 sigma must give size >= 10%.
(b) Tuple-level Mahalanobis Delta_J (5 gated tuples per window; covariance through linear
re-triangulation with fixed subset cameras, so camera uncertainty is NOT propagated; oracle sigma,
sigma_hat from the subset BA residuals, 0.5 sigma). kappa_max on the calibration half (pooled):
the largest of 10 log-spaced values between the 50th and 99th percentile of calibration kappa
whose calibration size(0.05) (sigma_hat) is in [4%, 6%]. On the evaluation half: size(0.05) in
[4%, 6%] at every sigma, every kept kappa decile in [3.5%, 6.5%], KS D <= 0.02, size(0.01) <= 2%,
retention >= 50%; misspecified control size >= 10% (else null calibration(b) INVALID). If ungated calibration
size is already in [3.5%, 6.5%] the gate is removed (ablation result).
(c) Conformal p-values (`EmpiricalNull`, (1 + #{ref >= s})/(n + 1)) for pencil_q10, asinh_q10,
acat, gauge_q90, registration_rms, sampson_mean, glrt_all: calibrated on the sigma = 0.5
calibration half; size(0.05) on the sigma = 0.5 evaluation half must be in [3.5%, 6.5%]
(exchangeable target); sizes at the other sigmas and on the shifted scenes are reported (expected
to fail: exchangeability is broken, shortlist R16). Mondrian conformal (bins = quartiles of
sigma_hat over the pooled calibration halves): size(0.05) in [3.5%, 6.5%] at every sigma.

### Clip aggregation, K, split, CUSUM, e-values - `clip_windows.py`, analysis `clip_analysis.py`
Clip = 40 points, 48 cameras on a 240-degree arc, 8 windows of 6 views. d* per family = smallest
injection grid dose with pencil_q10 power >= 0.5 at sigma = 0.5 (else 16 px). Calibration 400 single windows;
300 rigid clips; 100 clips each of jitter, drift, warp at d* (injected in every window); 100 change clips
(drift from window 4). Seeds 34,000,000 + task; clip point sets 43,000,000 + 1000 role + clip; calibration point sets 43,500,000 + i.
Window p-values: conformal from the calibration windows (pencil_q10, glrt_all, sampson_mean),
analytic for f_ratio; abstained windows p = 1. Aggregators: naive Fisher, CCT, raw HMP, Bonferroni,
block Fisher, e-value product (kappa = 0.5, reject at >= 20). An aggregator is eligible iff its
clip size is in [3.5%, 6.5%]; the V6 winner is the eligible one with the highest minimum power over
jitter/drift/warp. The e-product is valid iff 0 < size <= 6.5% (it is conservative by design; judged on power
at valid size). K* = smallest K in {100, 300, 1000, 3000, 10000, 50000} whose window power (drift at
d*) is within 0.01 of K = 50,000. Interleaved vs contiguous split and the U-statistic over the 10
balanced 3/3 partitions are descriptive. CUSUM on z = Phi^{-1}(1 - p): h calibrated on rigid clips
0-149 for ARL0 = 50 windows (the shortlist's 500 is not measurable with 2,400 null windows;
deviation), evaluated on clips 150-299 (pass: within +/-30% of 50); median delay after the change
<= 2 windows. e-process (anytime-valid): false-alarm rate on rigid clips <= 5%.

### Detector benchmark and decision - `detector_benchmark.py` on injection grid data
AUC per family x dose against the rigid test windows; paired DeLong pencil vs each statistic at d*.
A statistic's power/AUC counts only if its rigid test size is in [3.5%, 6.5%] (size before power).
- classical-test benchmark rule 1: pencil AUC < max(split_glrt, glrt_all) - 0.05 at every dose of every family jitter/drift/warp/swap
  -> the GLRT becomes S_alg and the pencil statistic is demoted to localisation.
- classical-test benchmark rule 2: pencil AUC >= that GLRT on at least one family at d* -> keep both.
- classical-test benchmark rule 3: Sampson within 0.02 AUC of the best size-valid statistic on every family at d*
  -> the claim "projective invariants add value" is dropped at the synthetic stage.
- **Decision:** if, on every family jitter/drift/warp/swap at d*, max(AUC Sampson, AUC GLRT) >=
  AUC pencil - 0.01, the framing becomes a methods paper on diagnostics and their limits.
Secondary (descriptive, no decision): asinh, ACAT, gauge fit, quantile variants, deltaj, sigma arms.
Prediction (exploration, 15 windows/cell): GLRT with known sigma and Sampson match or beat the
pencil statistic; the sigma-free F-ratio has almost no power because inconsistency also inflates
the within-subset residuals of its denominator.

### Representation robustness of the coplanarity score - `coplanarity_robustness.py`
80 fresh trials (seeds 35,000,000 + t), matched degeneracy experiment generator and M8 arm, 12 samples per k = 0..8
(8,640 samples), label k >= 6. Variants R0 (pre-registered representation; replication), R1 affine
X/X_4, R2 fixed PGL(4) frame per sample, R3 first-three base triple, R4 smallest triangle, R5 random
triple; new arm tyler_sigma6 (exactly frame- and representative-free; not noise-whitened).
H1_r: AUC(cop_r) - AUC(degensac_h_inliers), trial-cluster bootstrap 2000 (seed 35,900,000); 95% CI
for R0, 99% (Bonferroni over 5) for R1-R5. **The coplanarity claim ("beats the DEGENSAC inlier count")
survives only if every H1_r lower bound is > 0; otherwise it is withdrawn and only the
representation-specific statement may be reported, labelled as such.** H_T: tyler_sigma6 vs
DEGENSAC (95%); TOST of tyler_sigma6 vs homography_ls6 (90% CI within +/-0.01).
GRIC: already a baseline in matched degeneracy experiment (AUC 0.689); reported again here.
Prediction (exploration, 6 trials): R1 loses to DEGENSAC, tyler_sigma6 is uninformative
(AUC ~ 0.58), so the claim is expected to be withdrawn.

### Selection by information - `selection_by_information.py`
1000 pencil drift experiment-style drift scenes (seeds 38,000,000 + i); max coverage vs greedy D-optimal
(log det(1e-9 I + sum G_t^T G_t), k = 6) vs 6 random feasible tuples; downstream AUC/TPR of the
median-tuple drift score, coverage, log det. Descriptive only.

## What these experiments can and cannot show

All results are synthetic, for the generators above. Thresholds are calibrated on data from the
same generator; nothing here transfers to real images without the real-null go/no-go real null. A passed kill
switch shows that the statistic responds to injected point-level inconsistency, not that generated
media contain such inconsistency. A lost detector benchmark comparison is reported as such ("same data, same
ceiling"). No novelty is claimed for any arm (all are standard tools applied
here is open).

## Seeds (fresh seeds; registry additions)

Experiment IDs 21 (score audit permutations, seed 42021). Confirmatory bands: noise-model check 31,000,000-31,001,999 and
bootstrap 40,000,000; injection grid 32,000,000-32,005,099 and checks 32,900,000-32,901,199; null calibration
33,000,000-33,002,599; clip aggregation 34,000,000-34,005,999, clip point sets 43,000,000-43,005,099 and calibration point sets 43,500,000-43,500,399; representation robustness check
35,000,000-35,000,079 and bootstrap 35,900,000; selection-by-information study 38,000,000-38,000,999. Exploration (seen before
this file was written): 96,000,000-96,999,999 and smoke runs at +60,000,000 (92-94e6 band).

## Exploration before registration (motivation, not evidence)

`experiments/eval/explore_mv.py` (seeds 96,000,000 + i; `eval_exploration.csv`): m0 = 2.198e-4;
0.93 s per window; pencil, gauge, registration, Sampson and MAGSAC respond at 8 px (jitter/drift/warp) and to
10% swaps; f_ratio does not. Smoke runs of every runner and analysis on 92-97e6 seeds
(not committed). Those numbers are not quoted as results anywhere.

## Deviations

(Empty at registration. Anything changed after a confirmatory run is recorded here with the reason.)

### Deviation: detector benchmark decision rules: GLRT arm not size-filtered in code (found after the run, 2026-10-12)

Section 2 (detector benchmark) says a statistic's AUC counts only if its rigid test size is in [3.5%, 6.5%] (size before power).
`detector_benchmark.py` applied that filter to rule 3 but not to the GLRT term max(split_glrt,
glrt_all) of rules 1, 2 and the final decision. glrt_all has rigid test size 7.25% at sigma = 0.5
(outside the band), split_glrt 4.75% (inside). The text governs, so the primary reading uses
split_glrt only; the code's reading is reported next to it. No number changes; only which numbers
enter the rules. Values at d* (`detector_benchmark_cells.csv`, sigma = 0.5):

| Family (d*) | pencil_q10 | split_glrt | sampson_mean | glrt_all (size 7.25%) |
|---|---|---|---|---|
| jitter (2 px) | 0.9901 | 1.0000 | 1.0000 | 1.0000 |
| drift (4 px) | 0.9158 | 1.0000 | 0.9999 | 1.0000 |
| warp (2 px) | 0.8288 | 0.8382 | 0.7472 | 0.9880 |
| swap (0.05) | 0.9299 | 0.9204 | 0.8980 | 1.0000 |

- Step-9 decision: TRUE in both readings (methods-paper framing). In the text-compliant reading
  it is borderline on swap: 0.9204 >= 0.9299 - 0.01 = 0.9199 by 0.0005.
- classical-test benchmark rule 2 ("pencil >= GLRT on >= 1 family at d* -> keep both"): FALSE in the code's reading,
  TRUE in the text-compliant reading (swap: 0.9299 >= 0.9204). Primary: TRUE, keep both.
- classical-test benchmark rule 1: FALSE in both. Rule 3: FALSE in both.

### Deviation: null calibration crashed on a singular matrix; per-tuple guard added (2026-10-13, before any null calibration output existed)

The first confirmatory null calibration run stopped after 691 s with `numpy.linalg.LinAlgError: Singular matrix`
in the tuple-level Delta_J test (a path the 40-window smoke run did not reach) and wrote no file;
E4b then had no input. Fix (implementation only; no rule, grid, seed or threshold changed): a
numerical failure in one tuple now gives a row with `Error` and no p-value, and a failure before
the tuples are drawn gives one `Error` row for the window; window statistics are unaffected (they
were already guarded). E4b leaves the error rows out and reports their number. null calibration is re-run from
scratch on the same seeds. No null calibration number was seen before the fix.
