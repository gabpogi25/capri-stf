# Pre-registration C: Calibrated point trust scores (calibrated-trust experiment)

Status: first version written 2026-10-12, before any confirmatory run. Revision 1
(2026-10-12) answers the referees' blocking items. It is also written before any confirmatory
run (see "Revision log" and "Referee responses" at the end). No result on a seed in the
confirmatory ranges of Section 6 existed when either version was written. Any change after
the first confirmatory run goes in "Deviations" at the end, with date and reason.

## 1. Question

The point trust score R(p) (`reconstruction/certification.py::compute_point_trust_score`;
weights 0.5 conditioning, 0.3 drift, 0.2 descriptor; renormalised to 5/7 and 2/7
because rho is NaN within one reconstruction) was never fitted to outcomes
(negative finding N7). trust-calibration experiment reports ECE 0.212, but its label "Correct" means
"the scene is from the Generic regime". That is a regime label, not an outcome of
the item, and it is used for a score that trust-calibration experiment computes with a different formula
(L(S) and rho, not R(p)). **When the descriptor-free form R0(p) of the trust score (Section 5)
is mapped by a monotone calibrator fitted on a held-out split, does it give marginally calibrated
probabilities of a per-point ground-truth outcome on the synthetic distribution of Section 4?
Does it keep its ranking, and does it carry information beyond the base rate? How does it compare
with a raw kappa_J score? How badly does the calibration fail inside strata and under shift?**

## 2. Hypotheses

- **H1 (primary, composite).** On the test split, isotonic-calibrated R0(p) ("R-cal")
  (S1) has equal-width ECE <= 0.05, (S2) has lower ECE than uncalibrated R0(p) ("R-raw"),
  (S3) has an AUC that is not lower than R-raw's by more than 0.01, and (S4) has a
  positive Brier skill score over the constant base-rate predictor.
  S1 and S2 follow largely from fitting an isotonic map in-distribution (Section 9). They are
  necessary conditions, not evidence that R(p) is useful. The informative parts of H1 are S4
  and the AUC reported with it.
- **H2 (secondary, two-sided).** R-cal and K-cal (the calibrated raw kappa_J score,
  Section 5) differ in AUC or Brier score.
- **Pre-stated expectation (not a hypothesis).** From exploration (Section 10) we expect
  weak discrimination (AUC about 0.62) and a small skill score (BSS about 0.04). We also expect
  the calibration to be marginal only: ECE of about 0.3 inside the mismatch strata (Section 8),
  and under distribution shift an ECE of about 0.3 and a negative skill score.

## 3. Outcome label (positive class = trustworthy, Y = 1) and the item domain

**Item domain D.** Unit: one reconstructed point p that survives RANSAC and lies in at least one
accepted tuple (|A(p)| >= 1). Let A(p) be the accepted tuples of `run_capri_reliability_layer`
that contain p. These are exactly the tuples R(p) averages over. Points with A(p) empty
get R(p) = 0 from `compute_point_trust_score`, but they have no tuple, so they have neither a
label nor calibration evidence. They are excluded from calibration and test, counted, and
reported per split as a count and as a share of the RANSAC-surviving points (Section 8). The
calibration map is defined only on D. Outside D, `calibrated_trust` is None (Section 11), never
the probability of the lowest knot.

For a tuple t:
eps(t) = max_k |log|J_k(X^_t)| - log|J_k(X_t)||, k = 1..3. Here X^ is the reconstructed
cloud and X the generator's true world points. Both are whitened with
`whiten_homogeneous_points`, and J comes from `evaluate_six_point_invariants`.
eps(t) = +inf if the sign of any J_k differs, or if the reconstructed J is undefined.
**Y(p) = 1 iff median_{t in A(p)} eps(t) <= 0.1** (the worst invariant is within about 10%
relative error). If the true J is undefined, the item is excluded and counted
(expected count: 0).

Why this target. (a) R(p) claims to summarise "the algebraic quality of the correspondences
that contributed to its position". Its main term is a clipped function of kappa_J, and
kappa_J is the first-order noise gain of exactly these log-invariants
(docs/MATHEMATICS.md, Section 5). So eps measures the quantity R(p) is built to predict.
(b) The label is per item and comes from generator ground truth. It is not a regime label,
and no input of R(p) enters it. (c) J is a projective invariant, so the label needs no
alignment of the projective reconstruction to the truth. A fitted 15-dof homography would
put a learned step into the label. (d) It covers wrong correspondences: a mismatched point
is triangulated to a wrong position, so the invariants of every tuple that contains it are
wrong. We rejected "the point's correspondence is correct" as the primary label, because
with a constant descriptor confidence R(p) has no input that can see a mismatch that lies
on its epipolar line (rho = 0 within one reconstruction, Corollary 2.6). That label is still
reported (Section 8). The tolerance 0.1 was fixed in exploration (prevalence about 0.40, so both
classes have > 10,000 calibration items). It was not chosen on test data. Tolerances 0.05
and 0.2 are secondary.

## 4. Data generator (synthetic only; one scene = one seed)

The scene follows `run_synthetic_baselines.make_scene` with N = 60 world points. Draw order
(all from `np.random.default_rng(scene_seed)`): planar fraction f ~ U(0, 0.5); mismatch
fraction m ~ U(0, 0.2); pixel noise sigma_px = 10^U(log10 0.3, log10 3); round(N(1-f)) generic
points in [-3,3] x [-2,2] x [4,12]; plane normal; round(Nf) planar points (in-plane coefficients
U(-3,3), offsets U(-1e-3, 1e-3) x 8); a random permutation; view-1 noise; view-2 noise;
round(Nm) mismatch indices; then, for each mismatch, a depth factor U(0.6, 1.6) and noise.
Cameras: f = 800 px, 1280 x 960 px, R_y(0.3 rad), t = (-2, 0.2, 0.5). A mismatched view-2
point is the projection of the same world point scaled in depth along camera 1's ray. It
therefore lies on its true epipolar line, which models a repeated-texture match that
epipolar RANSAC cannot reject.

Pipeline (as in `main_pipeline.py`): `cv2.setRNGSeed(scene_seed)`; `findFundamentalMat(FM_RANSAC,
3.0, 0.99)` on raw pixels; `estimate_projection_matrices_from_fundamental`;
`triangulate_point_cloud`; `run_capri_reliability_layer(X, ones, conditioning_threshold=None`
(calibrated kappa_J median, 35.80), `conditioning_metric="kappa", max_tuples=200,
seed=tuple_seed)`; `assemble_point_certificates(X, certs, threshold)`.
**Tuple-selection seed.** tuple_seed = scene_seed + 600,000 (Section 6). It must not be the scene
seed: `run_capri_reliability_layer` builds `np.random.default_rng(seed)`, and with seed =
scene_seed that is the same PCG64 stream the scene generator used, so the tuple indices would
be a deterministic function of f, m, sigma_px, the permutation and the mismatch indices
(Revision log, item 1). In the real pipeline the tuple seed does not depend on the scene.
`np.random.default_rng` hashes its integer seed through `SeedSequence`, so the streams for
scene_seed and scene_seed + 600,000 are statistically independent. A check on 3,000 seeds in
the exploratory band gave a largest |correlation| of 0.064 between the first four scene draws and
the first 400 tuple draws. For n = 3,000 and 1,600 pairs, the expected largest |r| under
independence is about 0.07. With the shared seed it was 1.000. `cv2.setRNGSeed(scene_seed)` stays,
because OpenCV uses its own generator.
Descriptor confidence is 1.0 in the primary arm. We assume no descriptor model, so R(p) reduces
to its conditioning term, which is the part under test. Scenes where F fails, or with fewer than
8 inliers, are excluded and counted.
**Shifted test split (secondary):** m ~ U(0.2, 0.35) and sigma_px = 10^U(0, log10 5);
everything else is unchanged.

## 5. Scores, calibration and comparators (all fitted on the calibration split only)

- **R0 (the calibrated quantity).** The descriptor-free trust score
  R0(p) = (0.5 * mean_{t in A(p)} ell(t) + 0.2) / 0.7, where ell(t) is
  `compute_conditioning_signal(kappa_J(t), 35.80)`. This is `compute_point_trust_score` with every
  descriptor confidence set to 1 and the drift term dropped, as it is within one reconstruction.
  R0 lies in [2/7, 1] and is an affine function of the mean conditioning signal over A(p). So
  R-cal is a monotone function of mean ell alone. In the primary arm (descriptor = 1) the
  pipeline's trust_score equals R0.
- **R-raw:** R0(p) read directly as a probability. It is not a probability: it lies in [2/7, 1]
  while the prevalence is about 0.4. It is kept as the reference that N7 refers to, not as a
  serious competitor (Section 9).
- **R-cal:** `forensics.fusion.calibrate.Calibrator.fit(R0, Y)` (isotonic with >= 1,000
  items per class, otherwise Platt; the method used is reported).
- **K-raw / K-cal:** K(p) = -median_{t in A(p)} log10 kappa_J(t), taken from the
  certificates' `conditioning_score`. K-raw is used for AUC only (it is not a probability).
  K-cal uses the same Calibrator. When |A(p)| = 1, K and R0 are monotone transforms of each
  other (identical AUC). The comparison therefore tests whether R's linear clipping and
  its averaging over tuples lose information against log kappa_J.
- **Base:** a constant equal to the calibration-split prevalence (reference for skill).
- **Platt-R (sensitivity):** R-cal with Platt forced.

Fairness: every score sees the same test items and the same calibration items (all in D).
One calibrator class is used for all scores. Nothing is fitted or chosen on the test or the
shifted split.

## 6. Splits, seeds, sample sizes, power

S = get_experiment_seed(20) = 42020. Scene k has its own `np.random.default_rng(scene_seed)` for the
scene and its own `np.random.default_rng(scene_seed + 600,000)` for tuple selection.

| Split / stream | Scenes | Seed | Range |
|---|---|---|---|
| calibration (scene) | 600 | S + 70,000,000 + k | 70,042,020 - 70,042,619 |
| test (scene) | 1,000 | S + 71,000,000 + k | 71,042,020 - 71,043,019 |
| shifted test (scene) | 300 | S + 72,000,000 + k | 72,042,020 - 72,042,319 |
| descriptor arm draws | - | scene seed + 500,000 | 70,542,020-70,542,619; 71,542,020-71,543,019; 72,542,020-72,542,319 |
| tuple selection | - | scene seed + 600,000 | 70,642,020-70,642,619; 71,642,020-71,643,019; 72,642,020-72,642,319 |
| OpenCV RANSAC | - | `cv2.setRNGSeed(scene seed)` (separate generator) | as scene |
| cluster bootstrap | B = 2,000 | S + 73,000,000 | 73,042,020 |

Within C, the scene, descriptor, tuple and bootstrap ranges are pairwise disjoint (offsets 0,
500,000 and 600,000 are larger than the widest split, 1,000 seeds, and smaller than the
1,000,000 gap between splits). Overlap check (grep of all 8+ digit literals and `get_*_seed`
uses, 2026-10-12, confirmed by referees R1, R4 and R5): seed 42020 is unused. The largest
existing seeds are F7 at about 8.08e6 and the literals 20,240,611 and 20,261,011. The logistic fusion component
pre-registers 50,042,019 - 53,042,019 (plus bootstrap); the coplanarity component pre-registers 60,042,018 -
62,042,018. C therefore uses the 70-73e6 block, which is disjoint from all of them.

Exploration (Section 10) used scene seeds 90,042,020 + k (k < 600), shifted 90,043,020 + k
(k < 200), descriptor seed + 7 and bootstrap 91,042,020. The first exploration used the scene seed
for tuple selection. The decoupled re-run (revision 1) used tuple seeds scene seed + 600,000, that
is 90,642,020 - 90,642,619 and 90,643,020 - 90,643,219. The exploratory scene seeds lie inside the
exploratory band of B (90,042,019 - 90,048,018) and the bootstrap seed in A's (91,042,018 + ...); the
tuple seeds lie in neither. All of them are outside every confirmatory range.

Power (decoupled exploration: calibration 200 scenes, test 400 scenes, about 50 items per scene,
300 cluster resamples): SD of ECE_ew(R-cal) 0.0096 (point estimate 0.026, bootstrap mean 0.030);
SD of the Brier skill score 0.0073 (estimate 0.041); SD of Delta AUC (cal - raw) 0.0018
(estimate -0.0037). At 1,000 test scenes the SDs shrink by sqrt(400/1000) = 0.63, so the expected
upper ECE bound is about 0.030 + 1.96 x 0.0061 = 0.042 < 0.05. The skill score has z of about 9,
and the AUC margin of 0.01 lies about 5.7 SD from the expected Delta AUC. Calibration: 600 scenes
give about 29,000 items and about 11,000 positives, so the Calibrator uses isotonic regression.

## 7. Primary metric and decision rule

**Primary metric:** ECE of R-cal on the test split, 10 equal-width bins on [0, 1]
(comparable to trust-calibration experiment and its 0.05 criterion). CIs are 95% percentile intervals from a
paired bootstrap that resamples test **scenes** (items within a scene share tuples, so
they are not independent). The same resamples are used for every score. The resampled
ECE is biased upward, which makes S1 conservative.

**Success** iff all of S1-S4 hold: S1 upper bound of ECE_ew(R-cal) <= 0.05; S2 lower
bound of ECE_ew(R-raw) - ECE_ew(R-cal) > 0; S3 lower bound of AUC(R-cal) - AUC(R-raw)
> -0.01; S4 lower bound of BSS = 1 - Brier(R-cal) / Brier(Base) > 0. This is an
intersection-union test, so no multiplicity adjustment is needed. S4 is required because
a constant predictor can pass S1 (in exploration Base had ECE 0.017, lower than R-cal's 0.026).
H2: report Delta AUC, Delta Brier and Delta ECE (R-cal - K-cal) with CIs. A claim is
made only where a CI excludes 0.

## 8. Secondary analyses (descriptive, with CIs; no claims beyond them)

Equal-mass ECE (10 bins) and reliability-curve data (per bin: mean prediction, observed
frequency, n, bootstrap CI) for R-raw, R-cal, K-cal, Base and Platt-R. Conditional
calibration of R-cal (ECE, AUC and prevalence by stratum): (i) p mismatched / another point
of p's tuples mismatched / clean; (ii) sigma_px terciles (cut points from the calibration
split); (iii) f below or above 0.25; (iv) |A(p)| = 1 versus > 1. The **worst stratum ECE** is the
largest ECE over the ten strata of (i)-(iv), with its scene-bootstrap CI. Shifted test split: ECE,
Brier, BSS and AUC of the calibration-split map. Tolerances 0.05 and 0.2, each with its own
calibrator. AUC of R-raw for the label "correspondence correct" (expected about 0.5). Descriptor arm,
which assumes a descriptor model (not a measured one): desc ~ U(0.4, 1.0) for true matches
and U(0.25, 0.7) for mismatches; we report R's AUC and its ECE after a calibrator fitted on that
arm's own calibration split. This arm shows only whether a calibrator absorbs a second input, not
how real SIFT confidences behave. Its calibrator is not the shipped map (Section 11).
Counts, per split: excluded scenes, RANSAC-rejected points, surviving mismatches, **points outside
D (A(p) empty; count and share of RANSAC-surviving points; calibrated_trust None)**, sign flips,
undefined invariants.

## 9. Claims, wording and negative results

**What S1-S4 can and cannot show.** S1 and S2 follow largely from construction: an isotonic map
fitted on 29,000 in-distribution items and evaluated on the same distribution almost always has a
small marginal ECE, and R-raw is not a probability (range [2/7, 1], prevalence about 0.4), so S2 is
close to tautological. Success on S1-S4 therefore shows marginal calibration on the Section 4
synthetic distribution and a small amount of skill. It does not show conditional calibration,
calibration under shift or calibration on real images.

**Pre-specified wording on success (S1-S4 all hold).** The report uses this sentence, with the
measured values and their 95% CIs filled in from `trust_calibrated_metrics.csv` and `trust_calibrated_strata.csv`:
"On the synthetic distribution of Section 4, isotonic calibration of the descriptor-free trust score
R0(p) gives marginally calibrated probabilities of the invariant-accuracy label (ECE x [CI]); the
score discriminates weakly (AUC x [CI]) and adds little over the base rate (Brier skill score x [CI],
base-rate ECE x); calibration does not hold conditionally (worst stratum ECE x [CI], stratum s) or
under shift (shifted-split ECE x [CI], BSS x [CI])." The ECE, the AUC, the BSS, the worst stratum
ECE and the shifted-split ECE always appear in the same sentence. Neither the report, the docstrings
nor the docs call the output "calibrated trust" without the qualifier "marginally calibrated on the
synthetic Section 4 distribution". The field name `calibrated_trust` is defined in its docstring
with that qualifier.
**N7 status on success:** "addressed for marginal calibration on synthetic data only; conditional
calibration, calibration under shift and real images remain open." N7 is not marked closed.

**Negative results:**
- S1 or S2 fails: "isotonic calibration does not give marginally calibrated trust on this
  benchmark"; N7 stays open.
- S3 fails: "calibration costs ranking"; the optional map is not recommended.
- S4 fails while S1-S3 hold: "calibrated R0(p) reproduces the base rate; R(p) carries no
  usable information about invariant accuracy". This is reported as a negative result even
  though the ECE target is met. N7 stays open.
- Large stratum or shifted-split ECE is reported as a limit in the sentence above: the map is valid
  only for the item domain D under the scene distribution of Section 4, and gives no guarantee for
  real images, other thresholds or other descriptor models. Its docstring says so.
- Every result goes to `trust_calibrated_*.csv` via `save_results`, whatever the outcome. trust-calibration experiment and
  its results are not modified. Its 0.212 is not directly comparable, because its label
  and score differ.

## 10. Exploratory results already seen (motivate the design; not evidence)

**Decoupled re-run (revision 1; tuple seed = scene seed + 600,000, as in Section 4).** 600 scenes on
90,042,020 + k (calibration 0-199, test 200-599), label without the sign rule. At tol 0.1:
prevalence 0.392 / 0.409; ECE_ew for R-raw 0.427, R-cal 0.026 [0.015, 0.052], K-cal 0.027,
Base 0.017; AUC for R-raw 0.624, R-cal 0.620, K-cal 0.622; Delta AUC (cal - raw) -0.0037
[-0.0070, -0.0003]; BSS 0.041 [0.027, 0.054]. Strata (R-cal): p mismatched prevalence 0.053,
ECE 0.34; another tuple point mismatched prevalence 0.079, ECE 0.31; clean prevalence 0.72, AUC 0.71,
ECE 0.33; sigma_px terciles ECE 0.11 / 0.035 / 0.14; f < 0.25 / >= 0.25 ECE 0.040 / 0.017;
|A(p)| = 1 / > 1 ECE 0.041 / 0.061. Shifted split (prevalence 0.095): ECE 0.31, BSS -0.089,
AUC 0.65. For "correspondence correct", R had AUC 0.502. The descriptor arm had AUC 0.66.
Points outside D: 0 of 9,636 / 19,921 / 7,422 (calibration / test / shifted); 86% of items had
exactly one tuple. The ECE upper CI bound (0.052) is above 0.05 at 400 test scenes. The power
calculation of Section 6 projects it to 1,000 scenes.

**First exploration (coupled streams; superseded).** Same seeds, but tuple selection used the scene
seed. Prevalence 0.40 / 0.415; ECE_ew R-raw 0.42, R-cal 0.022 [0.011, 0.046], K-cal 0.023,
Base 0.017; AUC 0.628 / 0.627 / 0.624; BSS 0.048 [0.036, 0.060]; clean-stratum ECE 0.31; shifted
ECE 0.32. The coupling moved the point estimates slightly in R-cal's favour (ECE 0.022 against
0.026, BSS 0.048 against 0.041). These numbers are kept only as a record.

## 11. Files, code contract and runtime

Created: `experiments/run_trust_calibrated.py`; `experiments/results/trust_calibrated_items.csv`
(per point in D: split, scene seed, tuple seed, R0, K, R_desc, Y, eps, strata), `trust_calibrated_metrics.csv`,
`trust_calibrated_comparisons.csv`, `trust_calibrated_reliability.csv`, `trust_calibrated_strata.csv`, `trust_calibrated_counts.csv` (per
split: the counts of Section 8, including points outside D with count and share),
`trust_calibrated_calibration_map.csv`; `tests/test_trust_calibration.py`. All are written with `save_results`.

**`trust_calibrated_calibration_map.csv`** has one row per isotonic knot (columns `x`, `y`) and constant
columns that record the population it was fitted on: `input_convention = "descriptor_free_R0"`,
`conditioning_metric = "kappa"`, `conditioning_threshold` (the value returned by
`calibrated_conditioning_threshold(metric="kappa")` in the run, expected 35.796...; written from the
variable, never typed), `label_tolerance = 0.1`, `method` (isotonic or Platt) and
`domain = "|A(p)|>=1"`. The script asserts before fitting that the threshold it used equals
`calibrated_conditioning_threshold(metric="kappa")`.

**`reconstruction/certification.py`** gets additive changes only:
- `TrustCalibration` (frozen dataclass: knots, `input_convention`, `conditioning_metric`,
  `conditioning_threshold`, `label_tolerance`). `predict(r0)` is monotone piecewise-linear
  `np.interp` with clipping, identical to isotonic predict. The docstring states the domain D, the
  input R0, the synthetic Section 4 scope, the shifted-split and stratum limits, and the qualifier
  "marginally calibrated on the synthetic Section 4 distribution".
- `load_trust_calibration(path, expected_threshold, expected_metric="kappa")`. `expected_threshold`
  is a required argument, because certification.py cannot import the pipeline (circular import).
  The loader asserts: each metadata column holds a single value; `input_convention ==
  "descriptor_free_R0"`; `conditioning_metric == expected_metric == "kappa"`;
  `conditioning_threshold` matches `expected_threshold` (relative tolerance 1e-9); the knots are
  finite, x increases strictly, y does not decrease, and y lies in [0, 1]. A mismatch raises
  ValueError with the two values.
- `compute_descriptor_free_trust_score(certificates, conditioning_threshold)`: R0 as in Section 5
  (accepted certificates, every descriptor confidence replaced by 1). It returns None when no
  certificate is accepted.
- `assemble_point_certificates(..., trust_calibration=None)`. With a calibration: (a) ValueError if
  `conditioning_threshold` does not match `trust_calibration.conditioning_threshold`; (b)
  `calibrated_trust = None` for every point with no accepted tuple; (c) ValueError if an accepted
  contributing certificate has a finite consistency residual and drift threshold (a drift term the
  map was never fitted with); (d) otherwise `calibrated_trust = predict(R0(p))`. The map is never
  applied to `trust_score` computed with real descriptor confidences: `trust_score` keeps the
  pipeline's descriptor confidences and is unchanged; `calibrated_trust` ignores them by design.
- `PointCertificate.calibrated_trust: float | None = None`.
- `export_certificates_to_dict(point_certs, include_calibrated=False)`: unchanged output by
  default. With True it adds `calibrated_trust` (null outside D).
The default output, `trust_score`, `trust_label` and `export_certificates_to_dict` are unchanged
when no calibration is passed. The knots are read from the CSV and never typed into code.
`pipeline/main_pipeline.py` is not changed by this component. Wiring the map into the pipeline is
a separate decision for the author, and it must pass the calibrated kappa threshold.

**Tests** (`tests/test_trust_calibration.py`): None outside D; R0 does not depend on descriptor
confidences; equality of R0 and trust_score at descriptor 1; ValueError on a threshold mismatch, on
wrong metadata at load and on a finite drift term; knot monotonicity checks; unchanged default
output; predict equals sklearn isotonic predict on fitted data.

**Runtime.** The decoupled exploration ran 800 scenes in about 13 + 25 + 13 s in three parallel
processes (about 60 ms per scene in each process). The confirmatory run of 1,900 scenes, run
serially, takes about 2 min, plus the bootstrap (2,000 resamples of about 50,000 items; about 1.5
min). That makes about 3.5 min (target < 5 min).
**run_all integration (planned; confirmed with measured runtime at implementation).** An entry
`run_trust_calibrated.py` goes after `run_trust_calibration.py` in
`experiments/run_all.py`.

## Revision log

Revision 1, 2026-10-12, before any confirmatory run. One entry per blocking item.

1. **Tuple-selection stream coupled to the scene stream (R1 item 1, R5).** Accepted. Tuple
   selection now uses `seed = scene_seed + 600,000` (Sections 4 and 6, new rows in the seed table:
   70,642,020-70,642,619; 71,642,020-71,643,019; 72,642,020-72,642,319). `cv2.setRNGSeed(scene_seed)`
   is unchanged. Independence check in Section 4 (largest |r| 0.064 against 1.000 for the shared
   seed). The exploration was re-run with the decoupled seed on exploratory seeds only. Section 10
   now reports the decoupled numbers first and marks the coupled ones as superseded, and the power
   calculation of Section 6 uses the decoupled SDs (expected ECE upper bound 0.042 instead of 0.037;
   AUC margin about 5.7 SD instead of 15 SD).
2. **Calibration map used outside its population: empty A(p) (R1 item 2, R2 C-1(a)).** Accepted.
   Section 3 defines the item domain D (|A(p)| >= 1). `calibrated_trust` is None outside D, never
   the lowest knot's value (Section 11 (b)). The count and share of points outside D are reported per
   split (`trust_calibrated_counts.csv`, Section 8). The domain is stated in the docstring and in this document.
3. **Calibration map used outside its population: descriptor confidences (R2 C-1(b)).** Accepted.
   The calibrated quantity is now named R0, the descriptor-free form (Section 5), which equals the
   primary arm's trust_score. `assemble_point_certificates` applies the map to R0 recomputed with
   descriptor 1 (mean ell over A(p)), never to trust_score computed with real descriptor confidences.
   It refuses with a documented ValueError on a threshold mismatch or a finite drift term.
   `trust_calibrated_calibration_map.csv` records the convention, the metric kappa and the threshold (35.80),
   and `load_trust_calibration` asserts at load that they match (Section 11).
4. **Claim wording and N7 status (R4).** Accepted. Section 9 now says what S1-S4 can show, states
   that S1 and S2 follow largely from in-distribution isotonic fitting and that S2 is close to
   tautological, and fixes the sentence used on success. That sentence gives AUC, BSS, the worst
   stratum ECE and the shifted-split ECE alongside the ECE, and also the base-rate ECE. It defines
   the N7 status as "addressed for marginal calibration on synthetic data only" and bans unqualified
   "calibrated trust". Section 2 marks S4 and the AUC as the informative parts of H1. Section 8 defines
   the worst stratum ECE and adds BSS on the shifted split.
5. **Seed-stream reuse (R5).** Same change as item 1. The seed table of Section 6 now lists every
   stream: scene, descriptor, tuple, OpenCV and bootstrap.
6. **Non-blocking (R5): run_all integration.** A planned entry, table and docs targets are added to
   Section 11. The author owns the files concerned and confirms them at implementation.

## Referee responses

No blocking item was rejected. Two points of detail:
- R2 offered "recompute R with desc = 1, or refuse" for real descriptor confidences. We recompute,
  because R0 is a function of the certificates alone and needs no descriptor input. That lets
  `calibrated_trust` be defined for every point in D whatever descriptor confidences the pipeline
  used. We refuse (ValueError) only where recomputation cannot reproduce the fitted population: a
  threshold mismatch or a finite drift term.
- R1 and R5 suggested either a +600,000 offset or `SeedSequence(scene_seed).spawn`. We use the
  offset, because its ranges can be listed and checked for overlap like every other seed in the
  repository. `default_rng` already passes each integer seed through `SeedSequence`, so the two
  streams are independent (check in Section 4).

## Deviations

Recorded 2026-10-12 by the builder, **before the confirmatory run** (no file `trust_calibrated_*` existed
in `experiments/results/` when implementation started; an earlier builder had been interrupted
before running anything). The decisions below were taken after smoke runs on exploratory seeds
only (scene seeds 90,042,020-90,042,079, 90,043,020-90,043,079 and shifted 90,044,020-90,044,049;
bootstrap 91,042,020; all outside every confirmatory range). No confirmatory seed was run before
this section was written, and nothing was changed after the confirmatory run.

- **D1. Calibration-split variance (R3 non-blocking note).** The scene-bootstrap CIs of Section 7
  hold every fitted map fixed: they resample test scenes only and leave out the variance of the
  calibration split. R3 offered "say so, or add a refit sensitivity check". We say so. The
  interrupted builder had drafted a refit check; it was removed before any confirmatory run
  because the runner must stay within CODEME's 350-line limit and this component may not add a
  file outside its owned list. S1 therefore holds conditional on the fitted map.
- **D2. Equal-mass bins keep ties together (implementation detail Section 8 left open).**
  Equal-mass bin edges are the weighted deciles of the predictions, and an item goes to the bin
  given by the number of edges strictly below its prediction, so tied predictions always share
  a bin (empty bins are possible). The first implementation cut the sorted items into ten
  equal-weight groups, which splits a run of tied predictions at arbitrary item order. On the
  smoke run it gave the constant base-rate predictor an equal-mass ECE of 0.064 against an
  equal-width ECE of 0.015, an artefact of the split. Isotonic outputs take few distinct values
  (R1 note), so the same artefact would affect R-cal. With the fix, a constant predictor has the
  same ECE under both binnings. Equal-width ECE (the primary metric) is unaffected.
- **D3. H2 decision level (R1 non-blocking note).** Section 7 said a claim is made where a 95% CI
  excludes 0. H2 has three comparisons, so a claim now requires the Bonferroni 98.33% CI to
  exclude 0. Both intervals are written to `trust_calibrated_comparisons.csv`. This only makes H2 stricter.
- **D4. Additions requested in referee notes (no change to any rule).** Reported in addition:
  the number of distinct values each prediction takes (`n_distinct_predictions`, R1); the paired
  difference ECE_ew(R-cal) - ECE_ew(Base) with its CI (R4); Fisher exact tests of the scene and
  item exclusion rates of each split against the calibration split, with a flag at p < 0.05 (R1);
  the AUC of K-raw. The runner passes the value of `calibrated_conditioning_threshold(metric="kappa")`
  explicitly to `run_capri_reliability_layer` and to `assemble_point_certificates` (R3, R4); this
  is the value `conditioning_threshold=None` resolves to, so Section 4 is unchanged in substance.
- **D5. Draw order of the descriptor arm (Section 6 gives only the seed).** From
  `default_rng(scene_seed + 500,000)`: one U(0.25, 0.7) draw per RANSAC inlier, then one U(0.4, 1.0)
  draw per inlier; a mismatched inlier takes the first, a true match the second. The tuple's
  descriptor confidence is the mean over its six points, as in `run_capri_reliability_layer`.
- **D6. Reliability-curve bins.** The equal-mass bins of the reliability curves are fixed at the
  point estimate (unit weights) and kept across bootstrap resamples, so each bin's CI refers to a
  fixed set of items; the equal-mass ECE itself recomputes its edges in every resample.

### What S2 can show (R1 non-blocking note)

S2 is close to certain by construction (R-raw has range [2/7, 1] against a prevalence near 0.4;
exploration gave ECE 0.42 for R-raw). Its passing is not informative. The primary evidence of H1
is S1, S3 and S4, and of these S4 (with the AUC) is the informative part (Section 2).

### Run record (2026-10-12, third builder; no deviation)

The second builder ran the confirmatory experiment once (results written 08:04 UTC) and was then
interrupted by a usage limit before reporting. The third builder reviewed the runner, the
certification changes and the tests against this pre-registration before opening any result file,
found no departure from Sections 3-11 or from D1-D6, and changed no code, rule or file. It then
re-ran `experiments/run_trust_calibrated.py` once on the same pre-registered seeds. All seven
`trust_calibrated_*.csv` files were identical to the first run apart from the `# Code commit` header line, so
the reported numbers are those of a deterministic, repeated run. Measured runtime: 470 s (first run)
and 405 s (re-run), serially; the estimate of Section 11 was about 3.5 min. The bootstrap over
about 120 statistics takes most of it.
