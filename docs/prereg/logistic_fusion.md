# Pre-registration B: logistic Confidence Fusion with informative inputs (logistic fusion experiment)

Status: first written 2026-10-12 and revised twice the same day (hard floor, C0 and
threshold rule; then the referee revision R1: invariance status of the inputs,
scope, floor wording, GOAL row 10 decision table, dataset cache policy, deployable
arm D). All revisions were made before any confirmatory run; see "Revision log".
No result on a seed in the confirmatory range below existed when this was written.
Any change after the first confirmatory run goes in "Deviations" at the end, with
date and reason.

**Scope of every claim in this document.** Synthetic data only, with the two
measurements A and B given in a common, known unit chart: the generator's chart,
in which `measure_with_gaussian_noise` adds isotropic N(0, sigma^2) noise to the
true unit representatives and no gauge is applied. A real system never has this
chart (Section 4.1). Results of the primary arm do not transfer to real
reconstructions as specified; arm D (Section 5.1) is the only deployable
configuration tested, and it is secondary and descriptive.

## 1. Question

Neural fusion predicts the majority class (MCC 0, ROC AUC 0.61; negative finding N3,
GOAL row 10 "not met"). Two of its three inputs carry no information: rho from
one point set is identically zero (docs/MATHEMATICS.md, Corollary 2.6), and
min(1, L/1000) is saturated at 1. **On synthetic data with A and B given in a
common, known unit chart, and given the hard floor (Section 4.2), does a
classifier fed well-defined, informative per-tuple inputs predict a ground-truth
"tuple is bad" outcome better than (i) floor + majority, (ii) floor + the best
single input and (iii) floor + the neural model's inputs refitted with the same model? And is a
learned combination needed, or does a closed-form OR gate with three thresholds do
as well (CODEME section 7)?** A secondary, descriptive question: how much of the
result survives when only inputs available to a real system are used (arm D)?

## 2. Hypotheses

All three hypotheses are stated for synthetic data, with A and B given in a
common, known unit chart (the generator's chart; see the scope note above).

- **H1 (primary).** Model M1 (Section 5) has higher test-split MCC than each of
  the comparators C0 (floor + majority), C1\* (floor + best single input) and C2
  (floor + neural-fusion inputs, same model class as M1).
- **H2 (key secondary; decides the CODEME section 7 justification).** M1 has higher
  test MCC than C3 (floor + closed-form OR gate).
- **H3 (secondary).** M2 (gradient boosting) differs from M1 in test MCC (two-sided).

No hypothesis is stated for arm D (Section 5.1); it is reported with CIs only.

## 3. Outcome label (positive class = bad)

The label comes from generator ground truth for each item and is fixed before
any input is computed. It follows run_ablation.py (Table 9), so the target is
the decision the CAPRI gates are meant to make. An item is (scene, two noisy
measurements A and B, one random six-tuple). Regime = REGIMES[k mod 6]:

| Regime | Scene | Bad iff |
|---|---|---|
| Generic | 12 generic points; B measures the same scene | never |
| Sparse-View | 8 generic points (the scene distribution of ablation study; visibility lists unused) | never |
| Near-Degenerate | 12 points; the first 6 lie within h of one plane, h = 10^U(-6,-3) | tuple holds >= 4 of those 6 |
| Coplanar | 12 points; the first 5 exactly coplanar (ablation study `_coplanar_scene`) | tuple holds >= 4 of those 5 |
| Mismatched | `generate_mismatched_configuration(12, 0.15)`: 1 point replaced in B's scene | tuple holds the replaced point |
| Non-Rigid | `generate_non_rigid_configuration(12, b)`, b = 10^U(log10 0.005, log10 0.1) | max affine displacement over the tuple's 6 points > 0.02 |

Why this target. (a) It is defined by the generator, not by any input, so it
cannot leak into the label (the label-leak failure). (b) ">= 4 coplanar points" means
that at least one bracket vanishes up to the noise. Every offset satisfies
h <= sigma = 1e-3 (MEASUREMENT_NOISE_SIGMA), so that bracket carries no geometric
signal. This is also the level at which kappa_J responds (exploratory fact (1) of
the brief), not the ">= 6 coplanar" label of true_synthetic_baselines.csv.
(c) The non-rigid amplitude varies, so both classes occur in that regime and the
task is not a lookup of the regime. We rejected the neural model's pose-recovery label: it uses
4 synthetic extra points and an 8-point solver, and labelled 94-96% of items
positive. We rejected an "invariant error vs truth" label: the pencil input is a
noisy estimate of that same quantity (Proposition 2.3), so the label would be
close to circular. Expected bad fraction (exploration, 6,000 items): 0.229 overall;
by regime 0 / 0 / 0.27 / 0.13 / 0.49 / 0.49.

Chart dependence of the label. The Non-Rigid label is defined by affine
displacements in the common chart. `generate_non_rigid_configuration` moves each
point independently, so the 18 displacement coordinates of a six-tuple include
components that a projective frame change between A and B would absorb (only
3n - 15 = 3 are frame-free for six points). A frame-free detector therefore cannot
reach every positive of this regime, and the label is not a frame-free quantity.
The Coplanar, Near-Degenerate and Mismatched labels are statements about the
scene, not about the chart. This limits arm D's attainable accuracy and is
reported, not corrected.

## 4. Inputs and hard floor, with the invariance status of each

Every input below is computed from measurements A and B as the generator returns
them: in the common unit chart, with sigma = 1e-3 isotropic noise on the true unit
representatives and no gauge. "Observable" in the first draft meant only "computed
from A and B, not from the truth". It did not mean frame-free, and only one of the
five inputs is frame-free (table below). Status labels follow docs/MATHEMATICS.md.

logistic-fusion features, all log10, from existing code only (no dependency on the coplanarity component):
1. kappa_J(A) (`compute_invariant_noise_gain`);
2. L(A) (`compute_conditioning_score`; same information as delta_min);
3. sigma_4/sigma_1(A) (`compute_singular_value_conditioning`);
4. Q_0.1 of the finite pencil discrepancies D_t (45 for six points) between A and
   B (`pencil_discrepancies`; the test of MATHEMATICS Section 4);
5. rho under the independent-estimation model: each of the 15 brackets comes from
   A or B, chosen at random (`estimate_brackets_independently`). It is clipped at
   1e-12 before the log, and the number of clipped items is reported (expected
   about 0: P = 2^-14; 0 of 6,000 in exploration).

### 4.1 Invariance status

| Input | Status under a gauge (A_X, lambda) of each measurement (Eq. 1.1) | Label | Available to a real system as specified? |
|---|---|---|---|
| Q_0.1 of D_t | Frame-free: every pencil cross-ratio is gauge-invariant, so each D_t is unchanged when A and B get independent gauges. | proved (Lemma 2.1) | Yes |
| kappa_J(A) | Chart-dependent. Computed in the generator's unit chart, where Proposition 5.1 applies (isotropic noise of known sigma on unit representatives). | Prop. 5.1 holds in this chart only (proved, first order) | No. The pipeline evaluates it in the Tyler-whitened frame, where the noise is neither isotropic nor of known sigma. |
| L(A) | Chart-dependent (not PGL(4)-invariant; `whiten_homogeneous_points` docstring). | definition | No (same reason) |
| sigma_4/sigma_1(A) | Chart-dependent (SVD of unit columns changes under GL(4)). | definition | No (same reason) |
| rho (independent estimation) | NOT invariant when A and B get independent gauges. By Eq. (1.1) a bracket taken from B gains det(A_B) prod lambda_k^B and one from A gains det(A_A) prod lambda_k^A. The three products in a Grassmann-Pluecker sum then carry different factors, so rho is nonzero even for rigid, noise-free data. | proved (direct from Eq. 1.1) | No. It is informative here only because the generator puts A and B in one common frame with common unit representatives, the hypothesis of Theorem 6.1. |

Two consequences, stated here so that no later section overstates them.
(a) In the common chart, rho also responds to displacement components that a
frame change would absorb (Section 3, "Chart dependence of the label"). A real
two-reconstruction setting cannot see those components. Part of rho's value on
this benchmark therefore comes from information a real system does not have.
(b) The inputs kappa_J, L, sigma_4/sigma_1 and rho, and so M1, M1-lin, M2, C1 rules
on those inputs, C2 and C3, are **not transferable as specified**. The primary
arm answers a question about the common, known unit chart only. Arm D (Section
5.1) replaces them with deployable versions and is reported descriptively.

neural-fusion inputs (comparator C2): `build_fusion_features_neural(A, t)` = [min(1, L/1000),
rho from one point set, bracket strength], on measurement A, in the same chart
(chart-dependent; the single-set rho is identically zero by Corollary 2.6).

### 4.2 Hard floor (CODEME section 7)

An item is rejected by the floor if, on A or on B, any of the following holds:
- (stratum; algebraic) `classify_stratum_by_rank(whiten_homogeneous_points(X), t, tolerance=1e-3)`
  is CRITICAL. This is the rule of `run_capri_reliability_layer` (whitened frame,
  default tolerance). Frame-free up to the iteration tolerance of the whitening:
  Tyler's shape matrix is equivariant (cited, Tyler 1987; stated in the
  `whiten_homogeneous_points` docstring), so the whitened unit points are the same
  for every gauge up to one orthogonal map, and the rank test on unit columns does
  not change under an orthogonal map.
- (undefined; algebraic) `evaluate_six_point_invariants` returns None, or kappa_J or
  any feature is non-finite. An exactly zero bracket stays zero under every gauge
  (Eq. 1.1, det(A_X) != 0, lambda_k != 0), so the None case is frame-free; the
  non-finite checks inherit the chart of the feature they test.
- (conditioning; noise-model-dependent, NOT algebraic) sigma * kappa_J(X) >= 1,
  with the known sigma = 1e-3 and kappa_J in the generator's unit chart. The
  threshold is a **convention motivated by Proposition 5.1**: it is the point where
  the first-order SD of the worst log-invariant, sigma kappa_J, reaches 1. It is not
  derived from that proposition, which is first order (hypothesis sigma kappa_J << 1)
  and says nothing at sigma kappa_J = 1; its recorded failure at sigma kappa_J of
  about 1 shows saturation (kappa_J over-predicts the error), not loss of
  information. The floor also evaluates kappa_J at the noisy measurement (a
  plug-in), while Proposition 5.1 concerns the true q; the plug-in is least
  reliable in exactly this regime. *Hypothesis (not tested here as a claim):*
  above the threshold the invariants carry little information about the label
  beyond what the floor decision itself encodes. The constant 1 was not tuned: no
  other value was tried.
  Because it uses the known sigma and the generator's chart, this part of the
  floor depends on the noise model; the stratum and undefined parts do not.

Floor-rejected items get P(bad) = 1 and the decision "reject" under every classifier
and comparator. They are excluded from fitting and calibration of the learned models,
and they are kept in every test metric. The learned score can only add rejections:
accept = floor passed AND P(bad) < threshold. In exploration the floor rejected
12.2% of items (stratum: 1 item; conditioning: the rest). Of these, 62% were bad.
It caught 33% of all bad items (89-94% of the bad Coplanar and Near-Degenerate
items) and rejected 6.0% of good items (4.5-8.2% per regime). Those good items
count as false positives for every rule, and their number is reported.

## 5. Models and comparators

- **M1 (primary):** logistic regression (scikit-learn) on the degree-2 polynomial
  expansion of the 5 standardised log features. Pipeline: StandardScaler ->
  PolynomialFeatures(2) -> StandardScaler -> LogisticRegression(C=1.0,
  class_weight="balanced", lbfgs, max_iter=5000). Why quadratic: the label is a
  union of mechanisms (degeneracy OR inconsistency), and a linear logit cannot
  express an OR. Also, the noise in the pencil statistic grows with kappa_J
  (to first order D ~ sigma kappa_J), which is a kappa x pencil interaction.
  The form was chosen in exploration (Section 10), not on test data.
- **M1-lin (secondary):** the same model without the polynomial expansion.
- **M2 (secondary):** HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
  max_leaf_nodes=15, class_weight="balanced", random_state=42019, no early stopping).
- **Probabilities and decision (all learned models):** fit on the floor-passing train
  items. Platt-recalibrate on the floor-passing calibration items (unweighted 1-D
  logistic, C = 1e6, on `decision_function`). The threshold maximises the MCC of the
  full decision (floor OR P >= threshold) on all calibration items, over the
  0.5%..99.5% quantiles (step 0.5%) of the calibrated probabilities. Decisions at
  threshold 0.5 are reported as secondary.
- **C0 (floor + majority):** reject iff the floor fires; otherwise accept (majority =
  good). The floor-free majority rule (always accept; MCC 0, AUC 0.5,
  AP = prevalence) is reported as C0-raw.
- **C1 single inputs:** floor OR (s * feature >= threshold), for each of the 5
  features. The sign s and the threshold (same quantile grid) maximise the MCC of
  that full rule on train ∪ calibration. **C1\*** is the input with the highest
  train ∪ calibration MCC, chosen before the test split is touched. The other four
  are reported.
- **C2 (neural-fusion inputs):** the neural model's 3 inputs through the M1 pipeline (secondary: through M1-lin).
- **C3 (closed form):** floor OR log kappa_J >= a OR log Q_0.1 >= b OR log rho >= c.
  (a, b, c) maximise the MCC of the full rule on train ∪ calibration, over a 40^3
  grid of the 0.50..0.995 quantiles of each input.

Why the comparison is fair: within an arm, every rule sees the same items, the
same inputs (computed from A and B, never from the truth) and the same floor, and
no rule sees the test split before it is frozen. The rule-based comparators fit
their few parameters on train ∪ calibration (9,000 items), which is at least as
much data as M1 gets. This makes the comparison conservative for the learned model.
Fairness here is internal to the common-chart benchmark; it says nothing about
transfer (Section 4.1).

### 5.1 Arm D: deployable inputs (secondary, descriptive, no hypothesis)

Arm D asks what remains when only quantities a real two-reconstruction system can
compute are used. It was added in revision R1, before any confirmatory run, and
has no exploratory results (Section 10), so no power statement is made for it.

- **Gauges.** Each of A and B gets its own independent gauge, drawn from the item's
  generator after all draws of Section 6: A_X = U diag(10^{u_1..u_4}) V with U, V
  Haar-orthogonal (QR of a 4 x 4 standard normal matrix, signs fixed by diag(R) > 0)
  and u_i ~ U(-1, 1), so cond(A_X) <= 100; lambda_k = s_k 10^{w_k} with s_k a random
  sign and w_k ~ U(-1, 1). Draw order: A_X for A, lambda for A, A_X for B, lambda
  for B. Then A^g = A_X(A) A diag(lambda^A), and likewise B^g.
- **Inputs (4, all log10, all frame-free functions of the measured point sets).**
  kappa_J, L and sigma_4/sigma_1 of `whiten_homogeneous_points(A^g)` (frame-free up
  to the whitening tolerance, by Tyler equivariance as in Section 4.2: the whitened
  unit points are fixed up to one orthogonal map, which leaves all three scores
  unchanged), and Q_0.1 of D_t between A^g and B^g (Lemma 2.1). rho is dropped:
  there is no common chart in which it is defined. The whitened-frame scores have
  no Proposition 5.1 noise interpretation, because the noise is not isotropic of
  known sigma in that frame.
- **Floor D.** Stratum and undefined parts as in Section 4.2, computed on A^g and
  B^g (both frame-free). Conditioning part: sigma * kappa_J^W >= 1 with
  kappa_J^W from the whitened frame and the same constant sigma = 1e-3. This is a
  convention only, kept so that the CODEME section 7 floor has a conditioning part;
  Proposition 5.1 does not motivate it in this frame.
- **Rules.** M1-D and M1-lin-D: the M1 / M1-lin pipelines on the 4 inputs. C0-D:
  floor D + majority. C1-D: each input with floor D; C1\*-D chosen on
  train ∪ calibration. C3-D: floor D OR log kappa_J^W >= a OR log Q_0.1 >= b
  (40^2 grid, same quantile range as C3). Fitting, recalibration and thresholds
  exactly as in Section 5.
- **Reported (CIs from the same paired bootstrap, no claims beyond them).** Test MCC
  of every arm-D rule; Delta MCC of M1-D vs C0-D, C1\*-D and C3-D; and
  Delta MCC M1 - M1-D, the cost of losing the common chart. Per-regime acceptance,
  with the Non-Rigid caveat of Section 3. Gauge check: the largest
  |log10 Q_0.1(A^g, B^g) - log10 Q_0.1(A, B)| over all items (expected below 1e-6;
  Lemma 2.1). Failure control for the check: the same difference for log10 L,
  which is chart-dependent and must change.
  A design check during R1 (50 generic items, one scratch generator with seed
  90,000,001 in the exploration band, tuple (0..5); a check of the code, not
  evidence for any hypothesis) gave a largest log10 difference of 3e-10 for
  Q_0.1, 9e-11 for whitened kappa_J, and 7.2 for raw L.

## 6. Data, sample sizes, power

| Split | Items | Item seed (S = get_experiment_seed(19) = 42019) | Seed range |
|---|---|---|---|
| train | 6,000 | S + 50,000,000 + k | 50,042,019 - 50,048,018 |
| calibration | 3,000 | S + 51,000,000 + k | 51,042,019 - 51,045,018 |
| test | 9,000 (1,500 per regime) | S + 52,000,000 + k | 52,042,019 - 52,051,018 |
| paired bootstrap | B = 2,000 | S + 53,000,000 | 53,042,019 |

Each item has its own `np.random.default_rng(seed)`. Draw order: tuple
(`random_six_tuple`), regime parameter (h or b), scene, measurement A, measurement
B, bracket sources, then the arm-D gauges (Section 5.1). The gauge draws come last,
so they do not change any earlier draw. Data generation is a separate step before
any fitting (CODEME section 8).

**Dataset file and cache policy.** The item table (seeds, split, regime, label,
the 5 primary inputs, the 3 neural-fusion inputs, the 4 arm-D inputs, floor and floor-D
decisions with their causes) is written to `logistic_fusion_dataset.csv` with `save_results`.
- The default invocation, `python experiments/run_logistic_fusion.py` with no
  flags (the one `run_all.py` will use), **always regenerates** all 18,000 items
  from the seeds and overwrites the file. It never reads an existing dataset.
- Fitting and evaluation use exactly the values written to the file: after
  writing, the runner re-reads it (`pd.read_csv(..., comment="#",
  float_precision="round_trip")`), asserts with `np.array_equal` that every
  numeric column equals the in-memory array bit for bit (pandas writes float64 as
  the shortest repr, which round-trips), and fits only from the re-read frame.
- Reuse of an existing file is possible only through an explicit opt-in flag
  (`--reuse-dataset`). A reuse run prints a warning, writes its outputs under
  names with the suffix `_reused` and puts "dataset reused" in the experiment
  name of every header. It is never used for committed results; committed
  `logistic_fusion_*.csv` files come from the default invocation only.
- Reproducibility check (with the implementation): two default runs give
  byte-identical `logistic_fusion_*.csv` files apart from the commit line of the header.
  Because the default always regenerates, this check exercises data generation,
  not only fitting.

Overlap check (grep of all `make_rng` / `default_rng` / `get_*_seed` uses on
2026-10-12; seed 42019 is unused): the largest existing seed is about 8.08e6
(F7: 42107 + 80 * 100000 + ...). Others: ablation study <= 944,008; conditioning experiment and 5 <= about
1.0e6; math_lab 777,001 and 913,337 (+ track index); fixed literals 9001, 12345,
424242, 20,240,611 and 20,261,011. All are below 50,042,019. Exploration used
90,042,019 + k (k < 6,000) and bootstrap seeds 91,042,019 + r, outside the
confirmatory range. The author must check this range against the other
components' new pre-registrations.

Power: in exploration with the floor (2,400 test items, 3 rotations), the
paired-bootstrap SD of Delta MCC was 0.009-0.010 vs C3, 0.010-0.011 vs C1\* and
0.020-0.021 vs C0. At n = 9,000 each SD shrinks by sqrt(2400/9000) = 0.52 (about
0.005, 0.006 and 0.011). The exploratory differences were 0.026-0.031 (vs C3),
0.053-0.082 (vs C1\*), 0.41-0.47 (vs C0) and >= 0.5 (vs C2). A true difference half
as large as the smallest of these (0.013 vs C3) still gives z of about 2.6
(power about 0.75). The other comparisons have z > 4 even at half the effect.

## 7. Primary metric and decision rule

**Primary metric:** MCC on the test split at the calibration-chosen threshold.
CIs are 95% percentile intervals from a paired bootstrap over test items
(B = 2,000; the same resample indices for all classifiers).

- **H1 supported** iff the lower CI bound of Delta MCC is > 0 for each of
  M1 - C0, M1 - C1\* and M1 - C2. Success requires all three, so this is an
  intersection-union test and needs no multiplicity adjustment.
- **H2 supported** iff the lower bound of Delta MCC (M1 - C3) is > 0. If the
  upper bound is < 0, C3 is better. Otherwise the result is "no difference shown".
- **H3:** report Delta MCC (M2 - M1) with CI. No claim unless the CI excludes 0.

## 8. Secondary analyses (descriptive; CIs, no claims beyond them)

For every rule: balanced accuracy, ROC AUC and average precision (a C1 rule uses
its feature as the score, floor items set to +inf; C0 and C3 have no score, so they get no AUC or AP),
Brier score and ECE (10 equal-width bins of calibrated P(bad)), with paired-bootstrap
deltas vs M1. Also: per-regime acceptance of good and bad items; M1 coefficients and
standardisation; ablations of M1 (without the pencil input, without rho, conditioning
inputs only); counts of floor rejections by cause and of clipped rho values.
Operating-point variant F1: add the pipeline's calibrated conditioning gate (kappa_J
below the median over training-split Generic items) to the floor, and report the
metrics again.
Arm D (Section 5.1): the same metrics for every arm-D rule, and the deltas
listed there. Floor-rejection counts are reported separately for the
algebraic parts (stratum, undefined) and the noise-model-dependent conditioning
part, for both floors.

## 9. Negative results, GOAL row 10 status and required wording

**Status of GOAL row 10 for every combination of outcomes.** H1 has three
components (M1 vs C0, vs C1\*, vs C2; each "holds" iff the lower CI bound of
Delta MCC is > 0). H2 has three outcomes (supported; no difference shown; C3
better). Arm D and H3 never change the status.

| Case | M1 vs C0 | M1 vs C1\* and vs C2 | H2 | Row 10 status |
|---|---|---|---|---|
| A | holds | both hold (H1 fully supported) | supported | **Met** |
| B | holds | both hold (H1 fully supported) | no difference shown, or C3 better | **Partly met** |
| C | holds | at least one fails | any | **Partly met** |
| D | fails | any | any | **Not met** |

The cases are exhaustive (every one of the 2^3 x 3 outcome combinations falls in
exactly one row). The status always carries the scope qualifier below; "Met" is
never written without it.

**Required sentence for row 10** (the author copies it with the numbers
filled in; any omitted clause makes the row invalid):

> "<Met | Partly met | Not met> (synthetic data only, with A and B given in a
> common, known unit chart; not tested on real images or on independently gauged
> reconstructions). On the pre-registered logistic fusion experiment test split (9,000 items), the
> learned fusion M1 vs floor + majority (C0; the floor is part of the comparator,
> so this is not the plain majority baseline): Delta MCC = x [l, u]; vs floor +
> best single input (C1\* = <input>): x [l, u]; vs floor + neural-fusion inputs (C2):
> x [l, u]. H1 <is supported | fails vs <comparators>>. H2, vs the closed-form
> three-threshold OR gate (C3): Delta MCC = x [l, u], <supported | no difference
> shown | C3 better>. Deployable arm D (independent gauges, no rho; descriptive):
> M1-D vs C0-D Delta MCC = x [l, u]. The logistic fusion is off by default."

Case-specific consequences (in addition to the sentence):
- **Case A.** The `logistic_fusion.py` docstring may say that a learned combination is
  justified on this synthetic, common-chart benchmark, and only there.
- **Case B.** A learned combination is not justified on this benchmark: a
  closed-form OR gate does as well (or better). The docstring and the report
  recommend C3 instead of M1. The sentence must not describe M1 as needed.
- **Case C.** The report states "fusion adds nothing over <the single best input
  C1\* = <input> | the neural model's inputs>" in those words. If C1\* is not beaten, the
  docstring recommends C1\* (one threshold plus the floor).
- **Case D.** "The learned score adds nothing over the hard floor." The row text
  keeps the existing neural-fusion negative result next to the logistic-fusion one.
- **Every case.** The logistic fusion stays off by default in `main_pipeline`, because the inputs
  of the primary arm (unit-chart kappa_J, L and sigma_4/sigma_1, common-chart rho)
  do not exist in a real reconstruction (Section 4.1). It is available only as an
  explicit opt-in in `logistic_fusion.py`. Switching any fusion on by default would need
  a separate pre-registered test of deployable inputs, which arm D does not
  provide (it is descriptive). M1-lin and H3 are reported in every case.

Other negative reporting:
- Every result goes to `logistic_fusion_*.csv` via `save_results`, whatever the outcome.
  logistic fusion experiment uses synthetic data only. No claim is made about real images, about
  independently gauged reconstructions or about regimes outside the table in
  Section 3. The six-point pencil limit (3 frame-free degrees of freedom) is not
  addressed here.
- If arm D's M1-D does not beat C0-D (CI includes or lies below 0), the report
  says that no benefit of fusion was shown with deployable inputs.

## 10. Exploratory results already seen (motivate the design; not evidence)

6,000 items on seeds 90,042,019 + k, with a 40/20/40 split rotated 3 times.
Without the conditioning floor: test MCC was 0.75-0.78 for M1 and 0.68-0.69 for
M1-lin. With the full floor of Section 4: M1 0.77-0.80, M1-lin 0.76-0.79,
M2 0.76-0.78, C3 0.75-0.77, C1\* 0.69-0.75 (Q_0.1 in 2 rotations, rho in 1),
C0 0.30-0.37, C2 0.00-0.28 (unstable, since its inputs are near-constant).
M1 test ROC AUC was 0.93-0.95. The single-set rho of the neural fusion was <= 2.2e-16 in every
item, and min(1, L/1000) had median 1.

## 11. Files and runtime

Created: `ml_models/confidence_fusion/logistic_fusion.py` (feature extraction for both
arms, the hard floor, M1 with recalibration), `experiments/run_logistic_fusion.py`,
`experiments/results/logistic_fusion_dataset.csv`, `logistic_fusion_metrics.csv`,
`logistic_fusion_comparisons.csv`, `logistic_fusion_regimes.csv`, `logistic_fusion_model.csv`,
`tests/test_logistic_fusion.py`. Not modified: `train_neural.py`, `run_fusion_metrics.py`
and the exp8 results. No PyTorch (it is not installed).

**Required content of the `logistic_fusion.py` module docstring** (besides the CODEME
section 7 justification, which follows the Section 9 case): (1) the invariance
table of Section 4.1 in short form: Q_0.1 frame-free (Lemma 2.1); kappa_J, L and
sigma_4/sigma_1 chart-dependent, computed in the generator's unit chart; rho under
independent estimation not gauge-invariant (Eq. 1.1) and informative only with A
and B in one common chart with common unit representatives (Theorem 6.1's
hypothesis); (2) the sentence "Results apply to synthetic data with A and B given
in a common, known unit chart; the primary-arm inputs are not transferable to a
real reconstruction as specified"; (3) the floor's two kinds of parts: stratum and
undefined are algebraic; conditioning (sigma kappa_J >= 1, known sigma) is a
convention motivated by Proposition 5.1 and depends on the noise model; (4) that
The logistic fusion is opt-in only. A test checks that the docstring contains the phrases
"common, known unit chart", "not transferable" and "convention".

**Tests planned** (`tests/test_logistic_fusion.py`): gauge invariance of the Q_0.1
input on a fixed item (and, as failure control, that log L changes); the floor
never lets a learned score accept a floor-rejected item; the default runner path
regenerates rather than reads (it does not open an existing dataset file); the
round-trip assertion of Section 6 on a small item set; the docstring phrases above.

Runtime (estimate, to be measured and reported with the implementation): about
6 ms per item for the primary arm (two whitenings dominate), plus about 5 ms for
the two whitenings of the gauged measurements in arm D, for 18,000 items (about
3.5 min), plus fitting and bootstrap for both arms: about 5 min in total (target
< 8 min). The `run_all.py` integration (where the entry goes, measured runtime,
which table or figure, which docs and GOAL rows change) is specified with the
implementation, as R5 asks.

## Revision log

**R0 (2026-10-12, before any confirmatory run).** The first draft had a floor of
the stratum rule in the raw chart, undefined invariants and non-finite values.
That floor rejected 0 of 6,000 exploratory items, so it was inactive and contained
no conditioning part, against the brief ("the conditioning/stratum hard floor").
R0 applied the stratum rule in the whitened frame, as the pipeline does, and added
the conditioning floor sigma kappa_J >= 1 (R0 called it "theory-fixed"; R1
replaces that wording, see below). C0 became floor + majority. Thresholds maximise
the MCC of the full decision, including the floor. Power and Section 10 were
recomputed on the exploratory seeds. No confirmatory seed had been used.

**R1 (2026-10-12, referee round; before any confirmatory run; no new exploratory
run).** Changes, by blocking item:
- *B-1 (R2, invariance status).* Section 4 heading "observable only" removed.
  New Section 4.1 gives each input its invariance status and label: Q_0.1
  frame-free (Lemma 2.1); kappa_J, L, sigma_4/sigma_1 chart-dependent and computed
  in the generator's unit chart; rho under independent estimation not invariant
  under independent gauges (Eq. 1.1), informative only through the common chart of
  Theorem 6.1, and sensitive to frame-absorbable displacement. A scope note was
  added at the top; Question, H1-H3 and the GOAL row 10 sentence are now scoped
  to "synthetic, with A and B given in a common, known unit chart". Section 4.1(b)
  states that the primary-arm inputs are not transferable as specified. Section 3
  now notes that the Non-Rigid label is chart-defined. A deployable secondary
  arm D (Section 5.1: independent random gauges, whitened-frame scores, Q_0.1, no
  rho) was added; it is descriptive, has no hypothesis and cannot change the row
  status. The `logistic_fusion.py` docstring requirements were added (Section 11).
  Draw order gains the gauge draws at the end, so no earlier draw changes.
- *B-2 (R2, conditioning floor).* The floor is now called "hard floor" with
  labelled parts: stratum and undefined are algebraic (frame-free reasons given);
  conditioning is noise-model-dependent and not algebraic. The threshold is
  described as a convention motivated by Proposition 5.1 (the point where the
  first-order SD of the worst log-invariant reaches 1), not as derived from it.
  "Carries no usable information" was removed; a weaker statement is kept and
  labelled as an untested hypothesis. The plug-in issue (kappa_J at the noisy A,
  least reliable in this regime) and the saturation reading of the recorded V6
  failure are stated. The constant is unchanged.
- *R4 (row 10 status).* Section 9 now has an exhaustive decision table (cases
  A-D) covering every combination of the three H1 components and the three H2
  outcomes, including H1 fully supported with H2 failing (case B: "Partly met")
  and H1 holding vs C0 but failing vs C1\* or C2 (case C: "Partly met"). A
  required sentence names C0 as floor + majority, gives the three H1 components
  and the H2 outcome together, and carries the synthetic, common-chart scope.
  New: The logistic fusion stays off by default in every case (previously only if H2 failed),
  because its primary-arm inputs do not exist in a real reconstruction.
- *R5 (dataset cache).* Section 6 now fixes the policy: the default invocation
  (the one `run_all.py` uses) always regenerates the dataset; fitting uses only the
  values re-read from the written CSV, with a bit-for-bit round-trip assertion;
  reuse needs the opt-in flag `--reuse-dataset`, is labelled in file names and
  headers, and is never used for committed results.
Unchanged: label, primary inputs, M1/M2/C0-C3 specifications, seeds and splits
(confirmatory block 50,042,019 - 53,042,019), primary metric, decision rules of
Section 7, power analysis.

## Referee responses

All four blocking items are accepted; none is rebutted. Two clarifications:
- On B-1, we keep the primary arm in the common chart instead of moving it to a
  deployable setting. The question the brief asks (does fusion of informative
  inputs beat its comparators once the uninformative neural-fusion inputs are replaced?) is
  answerable there, and Exploration (Section 10) was done there. Changing the
  primary arm now would make the exploratory motivation and power analysis
  inapplicable. The cost is scope: every claim is limited to the common, known
  unit chart, and arm D shows descriptively what survives outside it.
- On B-2, we keep the conditioning part of the floor despite its non-algebraic
  status, because CODEME section 7 requires conditioning to act as a hard floor.
  Exploration showed that without it the floor was inactive (R0). Its label is now
  "convention", and floor rejections are reported by cause so that a reader can
  separate the algebraic from the noise-model-dependent rejections.

## Deviations

### D0. Implementation decisions and clarifications (2026-10-12, written before the confirmatory run)

All items below were fixed while implementing, after the referee round and
before any item in the confirmatory seed range (50,042,019 - 53,042,019) was
generated. A development dry run of the code used only exploration-band seeds
(items 90,042,019 + k and above, with outputs in the scratch directory, not in
the repository); nothing below was chosen from its numbers. None of these items
changes the label, the inputs, the models, the seeds, the primary metric or a
decision rule of Section 7.

1. *Regime order.* Regime = REGIMES[k mod 6] in the order of
   `run_ablation.REGIMES`: Generic, Near-Degenerate, Coplanar,
   Mismatched, Non-Rigid, Sparse-View (as in exploration). The table of
   Section 3 and the prevalence list "0 / 0 / 0.27 / 0.13 / 0.49 / 0.49" use
   the table order (Generic, Sparse-View, Near-Degenerate, Coplanar,
   Mismatched, Non-Rigid). Each split has exactly 1/6 of its items per regime.
2. *Sparse-View scene.* `generate_sparse_view_configuration(8, n_views=2,
   points_per_view=6)`, the ablation study call; its visibility lists are drawn and
   discarded. Exploration used `generate_generic_configuration(8)`. The point
   distribution is the same; the visibility draws only shift later draws in
   the item's stream.
3. *Gauge draws (arm D).* Within A_X: U, then V (each the Q factor of a 4 x 4
   standard normal matrix with diag(R) > 0), then u (4 uniforms). Within
   lambda: the signs (`rng.integers(0, 2, n)`), then w (n uniforms). lambda has
   one entry per point of the measurement (n = 12 or 8), because the whitening
   uses all points.
4. *Floor details.* On B the floor uses the same three parts as on A: stratum of
   `whiten_homogeneous_points(B)`, undefined (`evaluate_six_point_invariants(B, t)`
   is None or kappa_J(B) is non-finite), conditioning sigma kappa_J(B) >= 1.
   "Any feature non-finite" means any of the five log10 primary inputs (arm D:
   the four arm-D inputs). Floor D: stratum of the whitened gauged
   measurements, invariants of the raw gauged measurements, kappa_J of the
   whitened gauged measurements for the undefined and conditioning parts.
   The gauge-check control (Section 5.1) is log10 L of the raw gauged A^g
   against log10 L(A).
5. *Threshold grids and tie-breaks (R3, R5).* Learned models: candidate
   thresholds are the 0.5%..99.5% quantiles (step 0.5%, duplicates removed) of
   the calibrated probabilities of the floor-passing calibration items only;
   floor items (P = 1) are not in the grid but are in the MCC, which is always
   the MCC of the full decision (floor OR rule) on all items of the fitting
   split. C1 and C3: quantile grids of the floor-passing train ∪ calibration
   items, MCC of the full rule on all train ∪ calibration items. Ties: the
   first maximum wins, in the order lowest threshold first; for C1, sign +1
   before -1; for C3, lexicographic ascending (a, then b, then c); for C1\*,
   the first input in the order kappa_J, L, sigma_4/sigma_1, Q_0.1, rho.
6. *Floor items in the metrics (R1, R4).* As registered, floor-rejected test
   items enter every primary metric with P(bad) = 1 and decision reject.
   Added (secondary): every metric is also reported on the floor-passing test
   items only (`subset = floor_pass`), which separates the learned
   contribution from the floor.
7. *Scores for AUC / AP.* Learned rules: the calibrated P(bad), floor items 1.
   C1 rules: the fitted sign times the input, floor items +inf. C0-raw: a
   constant score (AUC 0.5, AP = prevalence). C0 and C3 have no score.
8. *C2 inputs.* the neural model's three inputs enter the M1 (or M1-lin) pipeline on their
   own scale, as in the neural fusion (they are bounded, not logged).
9. *Added secondary comparator (R2).* C2-norho: the neural model's inputs without the
   single-set rho (which is floating-point rounding, Corollary 2.6), through
   the M1 pipeline, so the M1 - C2 conclusion does not rest on scaled rounding
   noise. It does not enter H1.
10. *Variant F1.* "Report the metrics again" is implemented by refitting the
    learned rules (M1, M1-lin, M2, C2) and C1 / C3 with the F1 floor (floor OR
    kappa_J(A) >= the median of kappa_J(A) over all training-split Generic
    items). F1 is descriptive and enters no hypothesis.
11. *Bootstrap mechanics.* One `default_rng(53,042,019)`; the 2,000 resamples
    are drawn in 10 chunks of `rng.integers(0, 9000, (200, 9000))`. Metrics on a
    resample are computed from per-item multiplicities (weights), which equals
    recomputing them on the resampled items (tested against scikit-learn).
    **The CIs are conditional on the fitted models, the Platt maps, the
    thresholds and the gates: nothing is refitted inside the bootstrap, so
    training and calibration variability is not covered (R1).**
12. *Wording of H2 (R1).* C3 uses 3 of the 5 inputs. H2 is reported as a
    statement about this specific three-input OR gate; a win over C3 would not
    show that no closed form suffices. No five-input gate was added.
13. *H2 when H1 fails (R1).* H2 is reported and classified (supported / no
    difference shown / C3 better) in every case; it changes the GOAL row 10
    status only through the table of Section 9 (cases A and B).
14. *Output files.* Besides the five files of Section 11, the runner writes
    `logistic_fusion_hypotheses.csv` (H1-H3, arm D, row 10 case) and `logistic_fusion_counts.csv`
    (floor rejections by cause and split, clipped rho, prevalence, gauge check,
    v2 input checks, runtime). The dataset has `item_seed` and `regime`
    columns, and every file is written with `save_results(seed=42019)` (R5).
15. *Wording corrections (R2, R4; no effect on any computation).*
    (a) Section 3(b) is a heuristic: the bracket signal-to-noise ratio is about
    (h / sigma) times a configuration factor of order 1, so at h of about 1e-3
    it is of order 1, not zero; h and sigma are in different units. Exploratory
    fact (1) of the brief comes from 8-point samples with pixel noise and is
    motivation only; no result here confirms it.
    (b) The M1 rationale "D ~ sigma kappa_J" is withdrawn as stated: by
    Proposition 4.3(d), Q_0.1 follows the harmonic mean of the pencil scales,
    and the worst-conditioned invariants, which set kappa_J, barely move it.
    The quadratic form rests on the OR structure of the label. L(A) and
    sigma_4/sigma_1 are included because of the converse in Proposition 5.2:
    kappa_J can stay finite when one of the four brackets that J1-J3 do not use
    is zero.
    (c) Non-Rigid: of the 18 displacement coordinates of a six-tuple, 15 can be
    absorbed by a projectivity (3n - 15 = 3), so a frame-free input sees only
    the invariant part of a displacement above 0.02, while rho in the common
    chart sees all of it. Per-regime acceptance is reported for every rule,
    including the ablation M1-no-rho, so that a rho-driven gain in Non-Rigid is
    visible.
16. *Determinism (R5).* HistGradientBoosting uses OpenMP; the reproducibility
    check of Section 6 is run on the full default invocation (second run into a
    scratch directory, files compared byte for byte apart from the commit line).

### D1. After the confirmatory run (2026-10-12, second builder)

The first builder was interrupted by a usage limit after it had run the default
invocation once on the confirmatory seeds. Its outputs were in
`experiments/results/` when the second builder started. Items 1-7 below were
written after those results had been seen. No item changes a computation.

1. *Review and re-run.* The second builder checked the code against Sections 3-8
   and D0 and changed no computation. It made two tests stronger.
   (a) The test that floor items are never fitted now gives floor items finite
   inputs and inverted labels. It asserts that the fitted model and the Platt map
   equal those fitted on floor-passing items only. Before, a mutation that fitted
   on all training items passed it.
   (b) A new test checks the tie-break across signs and across the leading input
   of a two-input gate. Before, a mutation from `>` to `>=` in `fit_or_gate`
   passed every test.
   The default invocation was then run again. All seven `logistic_fusion_*.csv` files were
   byte-identical to the interrupted run's files, apart from the commit line of
   the header. The committed numbers are therefore those of the first
   confirmatory run, and nothing was chosen after seeing them.
2. *Docstring edits after the run.* After its run, the first builder wrote the
   outcome case (A) and the arm-D caveat of item 4 into the `logistic_fusion.py`
   docstring. The file time of `logistic_fusion.py` is later than its results.
   Item 1 shows that this edit changed no computation. The second builder added
   one docstring sentence about arm D (see the Reporting note below). The
   docstring cites the result files and contains no exp8b numbers.
3. *Runtime.* D0.14 put the runtime in `logistic_fusion_counts.csv`. The runner prints it
   instead, so that a re-run stays byte-identical. Measured with the C++ backend
   in this container: 285 s and 290 s wall time for two default runs, of which
   211 s was the generation of the 18,000 items. The target was < 8 min.
4. *The undefined part of floor D is not frame-free in practice.* Section 5.1
   called the undefined part frame-free ("an exactly zero bracket stays zero").
   As implemented (D0.4: `evaluate_six_point_invariants` on the raw gauged A^g
   and B^g), it applies an absolute tolerance of 1e-10 to the denominators of
   unit columns. A gauge with cond(A_X) up to 100 makes some denominators smaller
   than that, so the test fires.
   - Floor D: 352 of 18,000 items rejected as undefined (113 train, 67
     calibration, 172 test), 52 of them Generic, 224 of them good. Source:
     `logistic_fusion_counts.csv`, row `floor_d_undefined`; the per-regime and per-label
     breakdown comes from the columns `floor_d_undefined`, `regime` and
     `label_bad` of `logistic_fusion_dataset.csv`.
   - Primary floor: 0 items rejected as undefined.
   - kappa_J^W was finite for all 352 items.
   This was not changed, because arm D is descriptive. Every arm-D rule shares
   floor D, so the comparisons inside arm D stay paired and fair. Absolute arm-D
   MCCs, and Delta MCC M1 - M1-D, include these rejections.
5. *Cross-arm floor-pass deltas.* In `logistic_fusion_comparisons.csv`, the rows that
   compare rules of different arms (primary M1 vs D M1-D, primary M1 vs F1 M1)
   with `subset = floor_pass` are computed over different item sets: each rule
   uses its own arm's floor. Do not interpret them. Across arms, only
   `subset = all` is comparable.
6. *Reproducibility check (D0.16, R5).* A second default run wrote to a scratch
   directory gave byte-identical files apart from the commit line, M2 included.
   OMP_NUM_THREADS was not fixed, and no change was needed.
7. *Backend.* Every exp8b header records `Backend: cpp`. A re-run with the
   Python backend may differ in the last bits of kappa_J and the whitened
   inputs.
8. *Provenance of this document relative to the first confirmatory run
   (added 2026-10-12 after the final review, R1; text only, no re-run).*
   The first confirmatory run (the default invocation on seeds 50,042,019 -
   53,042,019) was made at about 03:20-03:28 UTC by the first builder, on commit
   2755b96 plus uncommitted files. Its reproducibility re-run into the
   scratch directory (files written 03:31-03:32 UTC)
   has the header "Code commit: 2755b96 (with uncommitted changes)", and its
   `logistic_fusion_hypotheses.csv` is byte-identical to the file committed before this
   item, apart from the commit line. The referee-revised Sections 1-11, the
   revision log R0/R1 and D0 were first committed in 3653699 at 07:41 UTC,
   about four hours after those results existed. The commit message of 3653699
   ("Written and revised before confirmatory runs") is therefore not supported
   by the repository record for this document, and an earlier note (B checker)
   that the result headers show the run came after that commit is wrong: the
   committed headers show 3653699 only because the later re-run of item 1 was
   identical. The statement that D0 and the revised sections were "written
   before the confirmatory run" rests only on the text of this document, on the
   B-designer scratch-file timestamps (up to about 03:04 UTC) and on 2755b96
   (02:27 UTC); no committed file records it. What 2755b96 does fix: the
   decision rule of Section 7 (H1 as an intersection-union test of the three
   lower CI bounds, H2 with its three outcomes, H3 report-only) is the same in
   substance as in 2755b96, and so are the label, the five primary inputs, the
   M1, M1-lin and M2 model specifications, the definitions of C1\*, C2 and C3, the
   seed block and the primary metric. What 2755b96 does not fix: the
   conditioning part of the floor, C0 = floor + majority and thresholds that
   maximise the MCC of the full decision including the floor (R0; in 2755b96
   the floor had no conditioning part and C0 was plain majority), arm D and the
   Section 9 table (R1), and every D0 item. No evidence was found that
   any of these was changed after the results, but their pre-run status cannot
   be shown from the repository. The integration commit must not describe this
   pre-registration as "written before confirmatory runs".

Reporting note (no change to the analysis). With deployable inputs (arm D),
M1-D did not differ from C1\*-D (the pencil input) or from C3-D: the CIs of both
Delta MCC include 0. Section 9 requires the report to say that no benefit of
fusion was shown only when M1-D does not beat C0-D. Here M1-D does beat C0-D,
but the report also states the C1\*-D and C3-D result, because it limits what
the primary-arm result means for a real system.

### D2. After the final review (2026-10-12, third builder; results already seen)

Everything in D2 was written after the confirmatory results had been seen. No
item changes the label, an input, a model, a threshold, a seed, the primary
metric or a decision rule, and no number in any result file changed (item 4).

1. *Correction of the rationale in D0.15(b) (R2; no effect on any computation).*
   D0.15(b) says that "L(A) and sigma_4/sigma_1 are included because of the
   converse in Proposition 5.2: kappa_J can stay finite when one of the four
   brackets that J1-J3 do not use is zero". This is right for L and wrong for
   sigma_4/sigma_1. sigma_4/sigma_1 is the smallest singular value of the whole
   4 x 6 unit-column matrix (`compute_singular_value_conditioning`), so it
   vanishes only when all six points are coplanar, not when one 4-bracket
   vanishes. The R2 counterexample (points 0, 1, 2, 5 exactly coplanar, so the
   unused bracket [1236] is zero) gives L = inf, kappa_J = 10.6 and
   sigma_4/sigma_1 = 0.37. The correct statement is: L sees this case;
   sigma_4/sigma_1 sees only near-coplanarity of the whole six-tuple. The same
   wrong clause was in the CODEME section 7 justification of `logistic_fusion.py` and
   has been corrected there. sigma_4/sigma_1 stays an input as registered
   (Section 4); only the reason given for it changes.
2. *Scope qualifier on the row 10 status (R4).* Section 9 says "Met" is never
   written without the scope qualifier, but `logistic_fusion_hypotheses.csv` had the row
   `GOAL row 10,case A,,,,Met` with no qualifier. The runner now writes the
   status as "<status> (synthetic data only, with A and B given in a common,
   known unit chart; not tested on real images or on independently gauged
   reconstructions)" for every case (constant `ROW10_SCOPE`), and a test checks
   this for cases A and D.
3. *Row 10 sentence: deployable arm and rho dependence (R4).* The required
   sentence of Section 9 reports for arm D only M1-D vs C0-D, a weak comparator.
   Two facts are added to it, and the author must copy them into GOAL row
   10 and the Table 8b text together with the registered clauses (a sentence
   without them is invalid, as for the registered clauses):
   (i) "With deployable inputs, M1-D was not shown to beat floor + the pencil
   input (C1\*-D: Delta MCC = x [l, u]) or the OR gate C3-D (x [l, u])."
   (ii) "Part of the primary-arm gain comes from rho, which is informative only
   through the common chart (M1 - M1-no-rho: Delta MCC = x [l, u])."
   These are descriptive and do not change the status (Section 9: arm D never
   changes it; the M1-no-rho ablation is a Section 8 secondary analysis). The
   runner now writes them to `logistic_fusion_hypotheses.csv` as rows `D-C1*`, `D-C3` and
   `rho` (comparisons M1-D - C1\*-D, M1-D - C3-D and M1 - M1-no-rho, subset
   all, MCC; decision "difference shown" iff the CI excludes 0). The values are
   the ones already in `logistic_fusion_comparisons.csv`; they are copied, not recomputed.
4. *Re-run.* The default invocation was run again after items 2 and 3. The six
   files other than `logistic_fusion_hypotheses.csv` are byte-identical to the previous
   files apart from the commit line. `logistic_fusion_hypotheses.csv` differs only by the
   three added rows and the qualified status string; its eight earlier rows are
   otherwise unchanged.
   The wall time of this run was 967 s (466 s of generation), against 285-290 s
   in D1.3, because other jobs shared the machine; the 8 min target of Section 11
   was met only on an unloaded machine.
5. *Row 10 sentence with the numbers filled in* (copied from
   `logistic_fusion_hypotheses.csv` of the run in item 4, rounded to 3 decimals; C1\* is
   the pencil input `log_pencil_q10`, `logistic_fusion_model.csv`):
   "Met (synthetic data only, with A and B given in a common, known unit chart;
   not tested on real images or on independently gauged reconstructions). On the
   pre-registered logistic fusion experiment test split (9,000 items), the learned fusion M1 vs
   floor + majority (C0; the floor is part of the comparator, so this is not the
   plain majority baseline): Delta MCC = 0.448 [0.427, 0.468]; vs floor + best
   single input (C1\* = pencil Q_0.1): 0.058 [0.047, 0.071]; vs floor + v2
   inputs (C2): 0.454 [0.433, 0.477]. H1 is supported. H2, vs the closed-form
   three-threshold OR gate (C3): Delta MCC = 0.022 [0.013, 0.031], supported.
   Deployable arm D (independent gauges, no rho; descriptive): M1-D vs C0-D
   Delta MCC = 0.437 [0.413, 0.460]. With deployable inputs, M1-D was not shown
   to beat floor + the pencil input (C1\*-D: Delta MCC = 0.006 [-0.005, 0.017])
   or the OR gate C3-D (0.004 [-0.005, 0.014]). Part of the primary-arm gain
   comes from rho, which is informative only through the common chart (M1 -
   M1-no-rho: Delta MCC = 0.038 [0.028, 0.049]). The logistic fusion is off by default."
