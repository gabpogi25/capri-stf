# Pre-registration A: degeneracy detection with matched inputs and a CAPRI coplanarity score (matched degeneracy experiment)

Status: written 2026-10-12 and revised the same day after referee round 1 (see "Revision
log" and "Referee responses" at the end). Both versions were written before any
confirmatory run. Git cannot show this: the file was first committed (3653699, 07:41) after the
confirmatory run (03:26), and the order rests only on this text and scratch-file timestamps (see
Deviations, "Provenance of this pre-registration"). No result on seeds in the confirmatory range below existed when this
was written; `experiments/results/` holds no `degeneracy_matched_*` file. Any change after the first
confirmatory run goes in "Deviations" at the end, with date and reason.

## 1. Question

Negative finding N5 / GOAL row 17 ("not met"): on 8-point samples labelled
degenerate when >= 6 of the 8 points lie on the dominant plane, capri_kappa had AUC
0.927 against DEGENSAC 0.961 (`true_synthetic_baselines_summary.csv`). The comparison had two
defects. (a) kappa_J measures how sensitive the invariants are to noise. It does not measure the size of the largest
coplanar subset, so it fires already at 4-5 coplanar points. (b) The inputs were not matched. CAPRI triangulated
with an F fitted by USAC-MAGSAC to all 1,000 scene correspondences, while DEGENSAC, GRIC
and the SVD condition number saw only the 8 sample points. **When every method sees
the same information, does a new CAPRI coplanarity (stratum-distance) score detect degenerate
samples better than DEGENSAC, and how does it compare with capri_sigma, capri_kappa, capri_ls,
GRIC, the SVD condition number and well-built continuous 2-D homography tests?**
The new score is not a conditioning score. Row 17 as worded ("CAPRI conditioning") is
answered by capri_kappa and capri_ls (H4), not by the new score (Section 6).

## 2. The new score (`algorithms/degeneracy/coplanarity.py`)

Coplanarity of a point subset Q is the vanishing of all brackets of Q, that is rank P_Q <= 3.
The stratum "the m points of Q are coplanar" has codimension m - 3 (Prop. 6.2).

**Primary score: the Sampson distance to the coplanarity stratum.** Let u be the
32 pixel coordinates of the 8 correspondences, and let X(u) be the CAPRI reconstruction from
them alone. X(u) is computed as follows: the linear 8-point F in Hartley-normalised coordinates (the charts T1, T2 are
the Hartley charts of the 8 observed points, computed once from the observed u and held fixed while J is differenced),
rank 2 enforced by Frobenius projection, P1 = [I|0], P2 = [[e2]x F | e2], DLT triangulation in the same charts, and
unit columns, with signs aligned to the unperturbed solution. For |Q| = m, take as base triple
(a, b, c) the three points of Q with the largest triangle in view 1. Use the m - 3
constraints g_d = [a b c d], d in Q \ {a,b,c}. Then

    T_Q = g^T (J J^T)^+ g / sigma_px^2,   J = dg/du (central differences, step 1e-3 px),

computed through the SVD of J with the rank and feasibility rules of Section 6.1 (never `np.linalg.solve`).
The sample score is `capri_coplanarity = -log10(min_{|Q|=6} T_Q)` (higher = more degenerate).
T_Q is the first-order (Sampson) approximation of the squared pixel distance, in
units of the noise, from the observed data to the set of data whose reconstruction **by this estimator in these
charts** makes Q coplanar (Sampson 1982; Hartley-Zisserman 2nd ed. §4.2.6; *cited*). sigma_px = 0.5 is the generator
value. DEGENSAC (2 px) and GRIC (sigma^2) also use it. sigma_px only rescales T, so the AUC does not depend on it.
In the scene arm the same formula is used with the scene cameras held fixed, and the DLT triangulation runs in the
scene's existing Hartley charts (computed over all 1,000 points by `projective_reconstruction`).

**Stratum estimate (secondary):** m_hat = max{m in 5..8 : min_{|Q|=m} T_Q <= chi2_{m-3}(0.99)},
or "<= 4". **Chart-dependent profile (reported, not primary):** delta_m = min_{|Q|=m}
sigma_4 of the unit columns of Q. This generalises `largest_coplanar_subset_size` (stratification.py) to n points and reuses
`normalize_to_unit_sphere`. capri_sigma, capri_kappa and capri_ls reuse
`compute_singular_value_conditioning`, `batch_invariant_noise_gain` and `compute_conditioning_score`
through `score_capri` / `score_capri_kappa` of `run_synthetic_baselines.py`, imported unchanged.

**What is invariant and what depends on the chart.** The module docstring of `coplanarity.py` carries this table
with the same rows and the same status labels (labels as in MATHEMATICS.md), and claims nothing beyond it.

| Property | Status |
|---|---|
| Q coplanar (all brackets 0) is invariant under GL(4) x (R*)^n, by Eq. (1.1) | proved |
| delta_m is invariant under per-point scaling and orthogonal changes of frame. Under A in GL(4) it changes by a factor in [1/kappa(A), kappa(A)], kappa = sigma_max/sigma_min, so its **magnitude depends on the reconstruction frame** | proved |
| delta_m is non-decreasing in m (column-deletion interlacing) | proved |
| **Counting.** The 8 correspondences give 32 numbers. The two-view model has 7 + 8 x 3 = 31 dof modulo PGL(4) (rank-2 F, then 3 dof per point given F). One residual dof remains: the 8 linear epipolar equations plus det F = 0 over-determine F, which is why the linear F of 8 generic noisy points has det F != 0. The reconstruction X(u) is therefore a **projection** of the data onto the 31-dimensional model, not a re-parametrisation. The discarded residual is a 1-dof check of the two-view model (the 8 points are not an exact fit) | proved (counting) |
| T_Q is unchanged to first order when g -> D(u) g with D smooth and invertible. This covers (i) a PGL(4) change of the reconstruction frame applied to the estimator's output (brackets scale by det A), (ii) per-point rescaling of the representatives (brackets scale by the product of the lambda_i), and (iii) any other non-collinear base triple, because both constraint sets have full rank m - 3 on the stratum (d[abcd]/dd = the normal of plane abc) and cut out the same smooth stratum locally. It does **not** cover the image-normalisation chart (next row). *Scope (post-hoc checker note, see Deviations):* the invariance is first order only and does not make the score representation-independent at finite noise, so capri_coplanarity is defined and claimed for unit columns and the largest-triangle base triple only. Backed qualitatively by `test_affine_representatives_change_distance_beyond_first_order`; no effect size on the matched degeneracy experiment samples is claimed | proved (first order), for (i)-(iii) only; beyond first order, (ii) fails numerically (in-project, qualitative only) |
| **T_Q depends on the image-normalisation chart (T1, T2) and on the estimator to first order.** The estimator (linear 8-point, Frobenius rank-2 projection, DLT) drops the residual dof of the counting row in a chart-dependent way, so the set "data whose reconstruction makes Q coplanar" differs between charts off the 31-dim model. Backed qualitatively by `test_distance_changes_with_image_chart` (Hartley chart vs an affine re-normalisation changes T by more than 1%). The R2 design check (exactly planar scene, k = 6, truly coplanar 6-subset: T_b/T_a = 0.9621, 0.9604, 0.9603 at sigma = 0.05, 0.005, 0.0005 px; near 1.15 at k = 4) came from a scratch script and is **not reproducible from repository code**; it motivated this row and is not cited as evidence. DLT triangulation with fixed cameras also drops one residual dof per point (4 numbers, 3 dof), so the SC version is chart-dependent too (not measured separately). **Hence capri_coplanarity is defined with the Hartley charts of the 8 observed points (M8) and with the scene's existing Hartley charts and fixed cameras (SC)**; no chart-invariance is claimed | numerical (in-project), qualitative only |
| For coplanar truth, Gaussian pixel noise and a fixed chart, T_Q -> chi2_{m-3} as sigma -> 0 (the truth lies on the chart's preimage set, which has codimension m - 3 where J has full rank) | proved (first order), per fixed chart; numerical check. Exploration: **heavy tail** in the 8-point arm (26% of true coplanar 6-subsets at k = 6 exceeded chi2_3(0.99)), so the chi2 thresholds are not trusted |
| Noise-free linear 8-point system: k <= 6 coplanar points (6 in general position on the plane, not on a conic) give rank 8 (unique F, via plane + parallax from the 2 off-plane points). k >= 7 gives rank <= 7, with a null space inside {[e]x H}. Every member of that null space reconstructs the k points as coplanar | proved; numerical check (exact planes: sigma_8/sigma_1 = 0 at k = 7, 8; 1.4e-4 to 3.8e-3 at k = 6) |

What this does and does not imply. By the data-processing inequality, no score computed from the same 32 numbers
(CAPRI's included) can beat the Bayes-optimal (likelihood-ratio) test of the label under the generator on those 32
numbers. This bound holds for every method and every outcome, so it predicts no ordering among the methods compared
here and is not used to explain any result. The reconstruction expresses the coplanarity of a subset (the zero set of
the brackets) in a form that does not depend on the PGL(4) frame, the representatives or the base triple; it does depend on
the image chart. The *score* T_Q inherits the frame, representative and base-triple invariance only to first order; at the
experiment's noise level it depends on the normalisation of the representatives (post-hoc checker note in
Deviations; shown qualitatively by a unit test), so every capri_coplanarity result is for unit columns and the
largest-triangle base triple.

## 3. Outcome label

Primary: **degenerate iff >= 6 of the 8 sample points are planar** (generator ground truth;
unchanged from N5 and Table 11, so the result answers GOAL row 17 and avoids outcome
switching). Why this is the right target: at k = 6 a homography explains 6 of the 8 points,
and F is fixed only by the parallax of 2 points. This is the H-degenerate case of DEGENSAC
(Chum, Werner, Matas 2005, stated for 7-point samples; *cited, not read in full*).
Secondary: **k >= 7**, the exactly rank-deficient case (Section 2).

## 4. Data, arms, methods

Generator: `make_scene` from `run_synthetic_baselines.py`, imported read-only (1,000 points,
20% within 1e-3 of one plane, f = 800 px, 1280 x 960, sigma = 0.5 px). Per trial: 9 values of
k (0..8) x 12 samples = 108 samples. Each sample takes k planar and 8 - k generic points, in that
draw order (as in the original). All scores are oriented so that higher = more degenerate.

**Arm M8 (primary, matched): each method sees only the 8 correspondences (plus sigma_px).**
capri_coplanarity, capri_sigma, capri_kappa and capri_ls use the 8-point reconstruction above
(capri_ls is added as a cheap descriptive method for the N5 sentence; it comes from the same `score_capri` call as capri_sigma).
The others are: degensac_h_inliers (`score_degensac`), gric_h_minus_f (`score_gric`) and svd_condition
(`score_svd`), all imported unchanged. Two continuous 2-D comparators are added,
because the integer inlier count has ties and a fair test should include strong 2-D baselines:
degensac_e6 = -min over 4-subsets of the 6th-smallest transfer error of the 8 points, and
homography_ls6 = -min over 6-subsets of the RMS transfer error of the DLT homography of those 6 points.
Both use `fit_homography_dlt` and `transfer_errors` unchanged (as `score_degensac` does).

**Arm SC (secondary, whole scene for everyone).** The scene F is computed with USAC-MAGSAC (`projective_reconstruction`,
as in the original). The scene H is computed with `cv2.findHomography(v1, v2, USAC_MAGSAC, 2.0, maxIters=20000,
confidence=0.9999)` after `cv2.setRNGSeed(seed)`. The methods are: capri_coplanarity (fixed scene cameras),
capri_sigma, capri_kappa and capri_ls (scene reconstruction), degensac_scene_count (sample points within 2 px of the scene H),
degensac_scene_e6 (minus the 6th-smallest transfer error under the scene H), and gric_scene (GRIC(F) - GRIC(H) on the 8
points with the scene F and H). svd_condition has no scene version and is listed as unchanged.
Why this arm: it gives the baselines the same scene-level information CAPRI had in Table 11.
Exploration: the scene H was sometimes wrong (46-55 inliers in 2 of 6 scenes), so this arm is not a ceiling.

Why the comparison is fair. In M8 all methods use identical inputs and the same samples. No method has a parameter fitted to
the test data. The only thresholds (5% FPR) come from the calibration split.

## 5. Seeds and sample sizes (S = get_experiment_seed(18) = 42018)

| Split | Trials (scenes) | Samples | Scene seed | Seed range |
|---|---|---|---|---|
| calibration (5%-FPR thresholds only) | 20 | 2,160 | S + 60,000,000 + t | 60,042,018 - 60,042,037 |
| test | 80 | 8,640 | S + 61,000,000 + t | 61,042,018 - 61,042,097 |
| paired cluster bootstrap | B = 2,000 | | S + 62,000,000 | 62,042,018 |

There is one `make_rng(seed)` per trial: scene first, then the 108 samples. cv2.setRNGSeed(seed) is called before each
scene-level USAC call. These calls do not use the numpy rng. One bootstrap rng (seed 62,042,018) draws the 2,000
resampled trial sets once; every CI below (all methods, both arms, both non-finite rules) uses those same sets.
Overlap check (2026-10-12): the existing experiments stay below
about 8.1e6 (logistic fusion pre-registration, Section 6). The synthetic baselines use 42,100-42,129. Prereg B uses 50,042,019-53,042,019, calibrated-trust pre-registration
70,042,020-73,042,020 (confirmed disjoint by R4 and R5). Exploration
for A used 91,042,018 + {0-15, 100-115, 200-211, 500, 700-705, 800, 900, 901}, and an
earlier interrupted attempt used 18,942,018 + {0-84}. All are disjoint from 60-62e6.

**Power.** Most of the variance is between scenes, so the design uses many trials with few samples
each. Exploration (36 scenes, 30 samples per k): AUC capri_coplanarity 0.981-0.986,
DEGENSAC 0.929-0.962, difference 0.029-0.034. The cluster-bootstrap SD of the difference,
rescaled to 80 trials with 12 samples per k, was 0.007-0.008, so z is about 4 at the explored effect (power about 0.98) and about 2 at half of it (power about 0.5).
H2 effects (0.017-0.32) have SD <= 0.005. For H3 the SD is about 0.0025, so the 90% CI half-width is about 0.004.
At an H3 difference near -0.003 the 90% CI can lie inside +/-0.01 while the 95% CI excludes 0; Section 6 fixes the wording for that case.

## 6. Primary metric and decision rules

**Primary metric:** the pooled ROC AUC over test samples in arm M8 for the label k >= 6. CIs are percentile intervals
from the paired cluster bootstrap over the 80 test trials (Section 5). Non-finite and rank-deficient values follow
Section 6.1, which was fixed before any confirmatory run.

### 6.1 Non-finite values and failed solves (each direction follows from what the value means)

Sentinels: MOST = +1e300, LEAST = -1e300 (data-independent; the runner asserts every finite score has |x| < 1e299, so
the sentinels sit above or below all finite scores and AUC, being rank-based, is unaffected by their magnitude).

**Subset level.**

| Score | Case | Primary rule | Meaning that fixes the direction | Sensitivity ("opposite") |
|---|---|---|---|---|
| capri_coplanarity | g or J has a non-finite entry | T_Q = NaN; the subset is left out of the min (nanmin) | the reconstruction failed, so the subset gives no reading | unchanged (NaN carries no direction; see sample level) |
| capri_coplanarity | J J^T rank-deficient: numerical rank r < m - 3, with r = #{s_i(J) > 1e-6 s_1(J)} (cond(J J^T) > 1e12); includes J = 0 | if the part of g outside range(J) has norm <= 1e-8 ||g|| (this includes g = 0): T_Q = g^T (J J^T)^+ g; otherwise T_Q = +inf, and it enters the min | the linearised problem min ||du|| s.t. g + J du = 0 is feasible iff g is in range(J); if feasible, the pseudo-inverse gives its exact minimum; if not, no first-order perturbation reaches the stratum, so the first-order distance is infinite (far from coplanar) | infeasible case: T_Q = 0 (treated as on the stratum) |
| capri_coplanarity | full rank | T_Q = sum_i (U^T g)_i^2 / s_i^2 from the SVD of J (never raises) | standard Sampson distance | unchanged |
| degensac_e6, homography_ls6 | a transfer error is +inf (the point maps to the line at infinity, or H has non-finite entries; `transfer_errors` already returns inf for NaN) | kept as +inf; it enters the 6th-smallest error / the RMS | the homography does not explain that point (least degenerate for that point) | the error is set to 0 (point treated as explained) |
| degensac_e6, homography_ls6 | rank-deficient DLT (numerical rank < 8 of the Hartley-normalised DLT matrix, s_8/s_1 < 1e-12) | not special-cased: the H returned by `fit_homography_dlt` is used, as in `score_degensac`; counted only | the returned H fits the subset exactly; its errors on the other points are what they are | unchanged |
| capri_kappa, capri_ls | non-finite kappa or L of a 6-subset | 1e300 (imported unchanged from `score_capri_kappa` / `score_capri`) | infinite noise gain = maximally ill-conditioned subset; kept unchanged so the N5 quantity is the one measured before | NaN -> left out of the max (no reading); +inf kept |
| capri_sigma | non-finite sigma ratio of a 6-subset | a NaN in any subset makes the sample value NaN (`np.min`, as in `score_capri`), which then follows the sample-level rule | the reconstruction failed, so the sample gives no reading | via the sample-level rule (NaN -> MOST) |

**Sample level (all scores, both arms).** +inf -> MOST (for capri_coplanarity, min_Q T_Q = 0: the data are on the
stratum; for svd_condition and capri_kappa, an infinite condition number or noise gain). -inf -> LEAST (for
capri_coplanarity, every subset is infinitely far from coplanar; for degensac_e6 and homography_ls6, no candidate
homography explains the points; these scores cannot be +inf, which the runner asserts). NaN (e.g. every subset of
capri_coplanarity is NaN) -> LEAST in the primary rule, because a detector that returns no reading raises no flag
(in a RANSAC loop such a sample is not rejected); MOST in the sensitivity rule. The rule of
`run_synthetic_baselines.py` line 267 (every non-finite value -> the largest finite value) is **not** used: for the
new scores it would call +inf transfer errors and infeasible subsets maximally degenerate, the opposite of what they mean.

**Counts and sensitivity.** `degeneracy_matched_summary.csv` gets, per arm, split, score and rule, the counts of: subsets with
non-finite g or J, rank-deficient feasible subsets (pseudo-inverse), rank-deficient infeasible subsets (+inf),
rank-deficient DLTs, infinite transfer errors, and samples mapped to MOST / LEAST from +inf, -inf and NaN.
A sensitivity block (`Rule = opposite`) recomputes every AUC, H1-H4 difference and D_sigma (Section 6.3) with the
opposite mapping. If any verdict differs between the two rules, the report gives both and says the verdict is not
robust to the handling of non-finite values. `tests/test_coplanarity.py` covers: a singular J J^T with g outside
range(J) (T_Q = +inf, no exception), a singular J J^T with g in range(J) (T_Q equals the pseudo-inverse value), a
non-finite g (subset excluded by nanmin; all-NaN sample -> NaN -> LEAST), and an H that maps a point to infinity
(transfer error +inf -> degensac_e6 and homography_ls6 move toward LEAST; all-infinite -> -inf -> LEAST).

### 6.2 Hypotheses

- **H1 (primary):** AUC(capri_coplanarity) - AUC(degensac_h_inliers) > 0 in M8.
  Supported iff the lower bound of the 95% CI is > 0. DEGENSAC is better iff the upper bound is < 0.
  Otherwise the result is "no difference shown".
- **H2 (key secondary, Bonferroni, 98.75% CIs):** capri_coplanarity beats each of capri_kappa,
  capri_sigma, gric_h_minus_f and svd_condition in M8 (lower bound > 0).
- **H3 (secondary; an empirical hypothesis motivated by exploration, Section 9, not by theory):** for X in
  {degensac_e6, homography_ls6}, D_X = AUC(capri_coplanarity) - AUC(X) in M8 lies within the equivalence margin
  +/-0.01. No theorem predicts that these 2-D tests equal T_Q or rank above or below it (they are not the Bayes-optimal
  test and do not use F). The report uses exactly one of these wordings:
  - 90% CI inside [-0.01, 0.01] and 95% CI contains 0: "equivalent within the +/-0.01 margin (difference D, 90% CI [..])".
  - 90% CI inside the margin and 95% CI excludes 0: "lower (or higher) by |D| (95% CI excludes 0), within the +/-0.01
    equivalence margin". The word "equivalent" is never used alone in this case.
  - 90% CI not inside the margin and 95% CI excludes 0: "lower (or higher) by |D| (95% CI [..])".
  - 90% CI not inside the margin and 95% CI contains 0: "inconclusive: neither equivalence nor a difference shown".
- **H4 (secondary, for the row-17 / N5 sentence; Bonferroni over 2, 97.5% CIs):** AUC(capri_kappa) - AUC(degensac_h_inliers)
  and AUC(capri_ls) - AUC(degensac_h_inliers) in M8. A conditioning score beats DEGENSAC iff its lower bound is > 0.

### 6.3 Scene-level dependence of capri_sigma (paired, decision rule)

D_sigma = AUC(capri_sigma, SC) - AUC(capri_sigma, M8), on the same 80 test trials and samples, with the paired
cluster-bootstrap 95% CI. M8 and SC differ in F (linear 8-point vs USAC-MAGSAC on 1,000 points) **and** in chart
(Hartley over 8 vs over 1,000 points, hence a different reconstruction frame), and capri_sigma is not PGL(4)-invariant.
This design does not separate the two. If the lower bound is > 0, the report says: "on these samples capri_sigma
scores higher with the scene-level reconstruction (its F and its chart together; this design does not separate them)
than with the 8-point reconstruction, so the Table 11 comparison gave CAPRI scene-level information the baselines
lacked". Otherwise it says "no dependence on the scene-level reconstruction shown". Table 11's 0.997 (other seeds,
42,100-42,129, in-sample) is quoted as context only and is never differenced against an matched degeneracy experiment number.
The same paired difference is reported for capri_coplanarity and capri_kappa as descriptive rows.

### 6.4 GOAL row 17 and N5

Row 17 reads "CAPRI conditioning beats a standard degeneracy test". It is marked "met" only if H4 shows a conditioning
score (capri_kappa or capri_ls) beating DEGENSAC. If only capri_coplanarity wins (H1), the row stays "not met" for
conditioning and the new result is added as a note; marking the row "met" for the new score would move the goalpost.
N5 ("L(S) loses to DEGENSAC") is reversed only if capri_ls itself beats DEGENSAC in M8 (H4 lower bound > 0).
The row-17 / N5 sentence is filled in from this template, with no clause dropped:

> Synthetic data only (`make_scene`, 8-point samples, label >= 6 planar). With matched 8-point inputs, a new CAPRI
> coplanarity score (a Sampson distance from the 8 correspondences to the coplanarity stratum of the 8-point
> reconstruction; not a conditioning score) had AUC a_c against a_d for the DEGENSAC inlier count (difference D_1,
> 95% CI [l, u]; H1 [supported / DEGENSAC better / no difference shown]). The CAPRI conditioning scores had AUC
> a_kappa (capri_kappa; difference to DEGENSAC D_kappa, 97.5% CI [..]) and a_ls (capri_ls = L(S); D_ls, 97.5% CI [..]).
> Against the continuous 2-D homography tests, capri_coplanarity was [H3 wording for degensac_e6] and [H3 wording
> for homography_ls6]. N5 ("L(S) loses to DEGENSAC") [is reversed / stands]. No real-image claim is made.

## 7. Secondary analyses (descriptive)

TPR on test at the calibration-split 5%-FPR threshold, with the realised test FPR. Also the in-sample
TPR at 5% FPR, for comparison with Table 11. The mean of per-trial AUCs. The flag rate at that threshold for each k
(0..8), and the AUC for k = 5 vs k = 6 only. All metrics for the label k >= 7. All of arm SC with
CIs. The m_hat confusion matrix against k, and the realised exceedance of chi2_3(0.99) by the true coplanar 6-subset
at k = 6. delta_6 (unit chart) next to capri_sigma. The counts and sensitivity block of Section 6.1.
**Runtime per method is printed to stdout only** and is never written to any `degeneracy_matched_*.csv` (nor are timestamps, host
names or thread counts), so a re-run reproduces the five result files byte for byte except the commit line of the
`save_results` header. The measured runtime is quoted from stdout in the run_all integration note.

## 8. Negative results and how they are reported

- H1 not supported: the report says the new CAPRI coplanarity score does not beat DEGENSAC when the inputs are matched.
  Row 17 and N5 follow Section 6.4 in every case.
- capri_coplanarity lower than degensac_e6 or homography_ls6 (H3, 95% CI excludes 0): reported as "a continuous 2-D
  homography test does at least as well on this benchmark", with the H3 wording of Section 6.2. No theoretical
  explanation is attached.
- D_sigma (Section 6.3): its lower bound decides between the two pre-written sentences; there is no other attribution.
- capri_kappa and capri_ls stay below DEGENSAC (H4): this confirms N5 and N5(a) with matched inputs.
- If capri_sigma is equal to or better than capri_coplanarity (H2 lower bound <= 0), the new score is called no improvement over capri_sigma.
- A verdict that changes under the opposite non-finite rule (Section 6.1) is reported as not robust.
  Every outcome is written to `degeneracy_matched_*.csv` with `save_results`. Synthetic data only. No real-image claim is made.

## 9. Exploration already seen (motivates the design; not evidence)

Candidates tried on exploratory seeds only: delta_m in the unit chart and sigma_4/sigma_1; Tyler-whitened delta_m.
The Tyler-whitened version fails by design, because Tyler's existence condition breaks at 6 of 8 coplanar points (Prop. 6.3 hypothesis).
Also tried: a GLS parallax-plane test, an F-constrained homography, a balanced chart, and the Sampson distance. On M8 the AUCs were:
Sampson 0.981-0.986, homography_ls6 0.983-0.990, degensac_e6 0.983-0.989, unit-chart delta_6 0.955-0.973,
DEGENSAC count 0.929-0.966, SVD about 0.89-0.90, kappa_J 0.84-0.88, GRIC 0.64-0.69.
On SC, Sampson and capri_sigma both had AUC about 0.998. The Sampson form, the base-triple rule and the step size were fixed from this exploration.
The exploration prototype used `np.linalg.solve` and had no non-finite rule; Section 6.1 replaces it and was not tuned on any data.

## 10. Files and runtime

Created: `algorithms/degeneracy/coplanarity.py` (module docstring carries the Section 2 table verbatim in substance,
including the counting row and the chart-dependence row), `experiments/run_degeneracy_matched.py`,
`tests/test_coplanarity.py`. Tests: exact coplanar data give T about 0; T unchanged to first order under a random PGL(4)
change of the reconstruction frame, under per-point rescaling and under a change of base triple; T **changes** under an
affine re-normalisation of the image chart on a planar k = 6 case (ratio not within 1e-2 of 1 at sigma = 5e-3 px; a
check that fails if the chart row were wrong); the linear 8-point F of 8 generic noisy points has det F != 0; the kappa(A)
bound and monotonicity of delta_m; rank 8 / 7 of the 8-point system at k = 6 / 7; consistency with
`estimate_projection_matrices_from_fundamental`; and the Section 6.1 cases. Results: `degeneracy_matched_samples.csv`,
`degeneracy_matched_summary.csv`, `degeneracy_matched_comparisons.csv`, `degeneracy_matched_by_planar_count.csv`, `degeneracy_matched_stratum.csv`.
Not modified: `run_synthetic_baselines.py` and its results. Each owned code file stays within 350 lines of functional
code; if the runner cannot, the author is asked before any further file is created.
Runtime: 10,800 samples. The unvectorised exploration code took about 25-45 ms per sample. With
batched DLT and batched Sampson solves the target is under 5 min. Parallelising over trials is allowed, because results are
seeded per trial and do not change. The run_all entry, measured runtime, target table and the docs / GOAL rows to change
are given with the implementation.

## Referee responses

All nine blocking items are accepted; none is rebutted. Two notes:
- A-1: the chi2 limit row is kept, restricted to a fixed chart. The R2 check shows the realised T_Q differs between charts,
  not that either chart loses the first-order chi2 law: for each chart the truth lies on that chart's preimage set of
  codimension m - 3. The heavy-tail caveat and "chi2 thresholds not trusted" remain.
- R2 A-3 and R4 item 7 ask for the same paired test with opposite signs. Section 6.3 uses D_sigma = AUC(SC) - AUC(M8), so
  "lower bound > 0" means the scene-level reconstruction scored higher (R4's decision rule; R2's attribution wording).

## Revision log

2026-10-12, referee round 1 (before any confirmatory run):
1. R2 A-1: Section 2 counting row rewritten (31 model dof vs 32 numbers, one residual dof, projection not
   re-parametrisation, "cannot check F" removed). Invariance row restricted to PGL(4) frame, representatives and base triple.
   New row: T_Q depends on the image chart and estimator to first order (numerical; R2 ratios 0.9621 / 0.9604 / 0.9603,
   reproduced) [annotation, final review: these ratios come from a scratch script and are not reproducible from
   repository code; Section 2 now cites the unit test, which checks the effect qualitatively]. The score is defined with the Hartley charts of the 8 observed points; SC chart stated. Module docstring
   bound to the same table; tests added for chart dependence and det F != 0.
2. R2 A-2: H3 restated as an empirical hypothesis; "theory predicts" and "Section 2 predicts this" removed; Section 8
   wording now "a continuous 2-D homography test does at least as well on this benchmark"; "adds no information" removed;
   Section 2 closing now "cannot beat the Bayes-optimal test on the same 32 numbers" and says this predicts no ordering.
3. R2 A-3 and R4 item 7: "well below 0.997" replaced by the paired D_sigma with cluster-bootstrap 95% CI and two
   pre-written sentences; attribution is to the scene-level reconstruction (F and chart together); Table 11's 0.997 is
   context only.
4. R3 and R4 item 8: new Section 6.1 with subset- and sample-level rules for every score, the SVD-based T_Q (no
   `solve`), the cond > 1e12 rank cut-off with a feasibility test, nanmin for non-finite subsets, the opposite-mapping
   sensitivity block, counts in `degeneracy_matched_summary.csv`, and the required tests. The line-267 rule is explicitly not used.
5. R4 item 5: Section 1 and new Section 6.4 name the score as a coplanarity (stratum-distance) score, add capri_ls to both
   arms, add H4 (capri_kappa and capri_ls vs DEGENSAC), fix when row 17 may be "met" and when N5 is reversed, and give
   the sentence template with the synthetic-only scope.
6. R4 item 6: H3 now has four exhaustive wordings, including "lower by X (95% CI excludes 0), within the +/-0.01
   equivalence margin"; "equivalent" is never used alone when the 95% CI excludes 0; the row-17 template uses them.
7. R5: runtimes go to stdout only, never into `degeneracy_matched_*.csv`; no timestamps or host data in result files.
8. Also: Section 5 states that one set of bootstrap resamples serves all CIs, and lists calibrated-trust pre-registration's range.

## Deviations

### Pre-run amendments (2026-10-12, implementation; written before the confirmatory run)

These follow the non-blocking referee notes or fix details the text left open. None was chosen
after seeing any confirmatory result, except that item P1 records an exposure.

- P1 (exposure, not tuning). While wiring the runner, two smoke tests ran on confirmatory seeds before the
  analysis code was final: (a) calibration trial 0 (seed 60,042,018) with 2 samples per k, printing
  per-sample scores; (b) calibration trials 0-2 and test trials 0-3 (seeds 60,042,018-020, 61,042,018-021)
  with 3 samples per k (the first 3 of the 12 pre-registered draws per k), printing pooled AUCs, CIs and
  verdicts on those 36 test samples. No threshold, parameter, method or decision rule was changed after this;
  the only later code changes were line-count compression, test fixes and a refactor that does not change
  any computed value. All later smoke tests used seeds outside every band (95,000,000-95,500,000).
  The confirmatory run below is the first and only run of the full design.
- P2 (R1, power). The three exploratory batches (8, 16 and 12 trials) gave H1 differences of 0.053, 0.024
  and 0.031 (R1's recomputation, m8_samp flipped), not "0.029-0.034". The rescaled SD 0.0078 is confirmed.
  At the smallest batch effect (0.024) z is about 3.1, so power is about 0.87, not 0.98.
- P3 (R1, R4). The DEGENSAC exploratory range is 0.929-0.962 (R1's recomputation) in both Section 5 and
  Section 9 (Section 9 said 0.929-0.966).
- P4 (R1, R5, non-finite mapping, made explicit per score). capri_coplanarity: min_Q T_Q = 0 -> +inf -> MOST;
  every subset infeasible (+inf) -> -inf -> LEAST; every subset NaN (non-finite g or J, or a failed SVD of the
  perturbed reconstructions, which marks all subsets non-finite) -> NaN -> LEAST (opposite: MOST); a singular
  J J^T is handled by Section 6.1 (pseudo-inverse if feasible, +inf if not). degensac_e6, homography_ls6,
  degensac_scene_e6: infinite transfer errors stay +inf (opposite: 0); a score of -inf -> LEAST; +inf is
  impossible (asserted by the sentinel check). A failed scene homography (cv2 returns None) is stored as a
  NaN matrix, so all its transfer errors are +inf (count 0, e6 = -inf -> LEAST). capri_kappa, capri_ls: the
  imported value 1e300 (score_capri / score_capri_kappa) is read back as +inf and mapped to MOST = 1e300,
  the same position; opposite for kappa: NaN subsets left out of the max. A non-finite reconstruction
  makes capri_ls, capri_kappa and capri_sigma NaN (sample rule). The +inf / -inf / NaN counts are reported
  by label class (calibration and test) in degeneracy_matched_summary.csv.
- P5 (R1, R3, R5, ties). Threshold = calibration negatives sorted ascending, element ceil(0.95 n) - 1
  (the rule of rsb.tpr_at_fpr); a test sample is flagged iff score > threshold (strict). The realised test
  FPR is reported next to each TPR.
- P6 (R1, H3). H3 is reported per comparator. "Equivalent to the continuous 2-D tests" (plural) is used only
  if both comparators get the first H3 wording.
- P7 (R2, Section 4 fact). make_scene puts planar points within 8e-3 world units of the plane
  (uniform(-1e-3, 1e-3) * 8 along the normal; about 1e-3 relative to the plane depth 8), not "within 1e-3".
  The rank-7 statement for k >= 7 holds for exact planes; the generator's k >= 7 samples are near rank 7.
- P8 (R2, chi2). The chi2 row needs the hypothesis that u -> g(X(u)) is smooth with J of full rank m - 3 at
  the noise-free data. It holds at k = 6 and fails at k >= 7. Stated in the module docstring (item 7).
- P9 (R2, step size). T_Q for m = 6 is also computed with step 1e-4 px in both arms. A sample "disagrees" iff
  the relative difference of min_Q T_Q between the two steps exceeds 1e-3 (or exactly one is non-finite).
  Counts by k go to degeneracy_matched_stratum.csv; a unit test requires agreement within 1e-3 at k = 0, 3, 5, 6.
- P10 (R4, N5(a)). Section 8's "confirms N5(a)" is replaced by a descriptive criterion: N5(a) ("kappa_J fires
  at 4-5 coplanar points", an exploratory observation) is called confirmed with matched inputs only if, in M8
  under the primary rule, the test flag rate of capri_kappa at its calibration 5%-FPR threshold for k = 4-5
  minus that for k = 0-2 has a 95% cluster-bootstrap CI with lower bound > 0. The same row is reported for
  capri_ls. "This confirms N5" (H4) keeps its Section 6.4 meaning.
- P11 (R4). The explore.py of the interrupted 18,942,018 attempt was overwritten, so its content cannot be
  recovered (as R4 noted).
- P12 (R4, R2). Status labels: the proofs of the "proved" rows are written in the module docstring of
  coplanarity.py. Row (4) proves "at least one" residual dof; "exactly one" rests on the cited standard
  count. Row (8) uses the cited HZ Results 13.1 and 13.3. No novelty is claimed for the score (the Sampson
  approximation applied to bracket constraints). capri_kappa and capri_sigma in M8 are evaluated in the
  8-point Hartley reconstruction chart, where pixel noise is not isotropic (docstring).
- P13 (R5). The sample size (20 + 80 trials, 12 per k) is fixed regardless of runtime. Trials run in worker
  processes with an ordered map and cv2.setNumThreads(1); results do not depend on the number of workers
  (unit test: one trial alone, twice, and inside a worker give identical rows).
- P14 (implementation details). (a) Arm SC recomputes the scene cameras with the same USAC-MAGSAC call and
  seed as projective_reconstruction (which returns points only) and asserts, per sample, that the
  fixed-camera DLT reproduces the scene points. (b) gric_scene re-implements the GRIC formula of
  rsb.score_gric for given models (score_gric fits its own); a unit test shows equality with score_gric
  when given the same models. (c) Added descriptive rows: neg_delta_6 (= -delta_6, unit chart) in both arms,
  the paired SC - M8 difference for capri_coplanarity and capri_kappa, m_hat in both arms.
  (d) The invariance unit tests (frame, per-point rescaling, base triple; with u-dependent factors applied
  inside every perturbed reconstruction, per R3) use sigma = 0.01 px on a well-conditioned configuration and
  require a relative change < 5e-3, with a negative control (||g||^2 changes by > 10%). A scratch check found
  that on a badly conditioned 8-point configuration the first-order regime is not reached even at
  5e-4 px (relative deviations 0.02-0.08; scratch figure, not reproducible from repository code),
  consistent with the heavy-tail caveat of the chi2 row.
  [Post-hoc annotation, checker round: "relative deviations 0.02-0.08" understates the dependence on the
  representatives at the experiment's noise level; see the post-hoc checker note at the end.]
  (e) The runner holds 350 functional lines (no extra file was needed).


### Post-run record (2026-10-12, after the single confirmatory run)

- The full design ran once (code at commit 2755b96 plus the uncommitted component-A files; the
  `save_results` header says "with uncommitted changes"). No analysis choice was changed after the run.
  After the run, only `tests/test_coplanarity.py` changed (the reproducibility test now compares frames
  exactly, `check_exact=True`); the runner and `coplanarity.py` did not change.
- Runtime: 307.8 s in total on 4 worker processes (sample collection 153.5 s; the analysis with the
  2,000-resample paired bootstrap the rest), above the 5-minute target. N was not reduced (P13).
- Reproducibility: test trial 0 and calibration trial 7, re-run alone in one process, reproduce their
  stored rows bit for bit (stored file read with `float_precision="round_trip"`).
- Non-finite values: none. Every count of Section 6.1 is 0 in both splits and both arms (no non-finite
  g or J, no rank-deficient J, no rank-deficient DLT, no infinite transfer error, no sample mapped to
  MOST / LEAST). The opposite rule therefore reproduces every primary number and verdict exactly.
- Step-size check (P9), descriptive: in M8, 69 of 8,640 test samples disagree between steps 1e-3 and
  1e-4 px (by k = 0..8: 5, 3, 3, 4, 6, 2, 10, 8, 28; 33 of 6,720 at k <= 6). In SC none disagree.
  A post-hoc diagnostic of the three largest disagreements (k = 0, 6, 6; scratch script, figures not
  reproducible from repository code) found 8-point design matrices with sigma_8/sigma_1 = 1.2e-4, 6.1e-5
  and 2.2e-4, where T_Q keeps changing as the step shrinks (the
  linearisation does not hold at these steps). The pre-registered step 1e-3 px is used for every
  reported number; this is reported as a limitation, not corrected.

### Post-hoc checker note (2026-10-12, after the run; scope of the claims, no verdict changes)

- Finding (checker; post-hoc). The checker applied the per-point rescaling of item (ii) of the Section 2 invariance
  row inside every perturbed reconstruction of the M8 test samples (affine representatives X/X_4 in place of unit
  columns, nothing else changed), and also a fixed PGL(4) frame change and other base triples. **Every figure in this
  bullet comes from scratch scripts (`gauge.py`, `triple_perm.py`, `gauge_tests.py`) that
  are not in the repository and wrote no `save_results` file; the figures are not reproducible from repository code
  and are not cited anywhere as evidence.** With affine representatives the AUC of capri_coplanarity fell from 0.9828
  (the stored value, which the scratch output matched) to 0.9153, below DEGENSAC's 0.9365, with a median absolute
  change in log10 T of 1.06. A fixed frame change (A = randn + 2I, then unit columns) gave 0.9818 (median change 0.07
  log10). Referee R3 reports that the checker's base-triple variants gave 0.9747 (first three points) and 0.9686
  (smallest triangle); the builder did not reproduce these two. On `noisy(seed, 6, sigma)` of
  `tests/test_coplanarity.py`, T_affine/T_unit at 1e-3 px was 1.000, 0.740, 0.932, 2.050 for seeds 6, 12, 29, 3.
- What is in the repository. `test_affine_representatives_change_distance_beyond_first_order` checks the effect
  qualitatively (affine representatives move T by more than 20% at 1e-3 px on seed 12, by less than 1e-3 on seed 6),
  so the test fails if the score became representation-independent at that noise level. No size of the effect on the
  matched degeneracy experiment samples is claimed.
- Consequence for the claims. The first-order invariance of item (5) / the Section 2 row is proved and stays proved,
  but it does not make the score representation-independent at sigma = 0.5 px. The score was pre-registered with unit
  columns and the largest-triangle base triple (Section 2), so the H1 verdict ("supported") and every number and
  verdict in `degeneracy_matched_*.csv` stand unchanged. What changes is the scope: every capri_coplanarity result is a result for
  that representation and base triple. The module docstring item (5), the Section 2 row and the Section 2 closing
  paragraph say this; P14(d) is annotated.
- The row-17 / N5 sentence (Section 6.4 template, filled in from `degeneracy_matched_comparisons.csv`; no clause dropped, one
  qualitative scope clause added after the H1 clause, with no scratch-only number):

  > Synthetic data only (`make_scene`, 8-point samples, label >= 6 planar). With matched 8-point inputs, a new CAPRI
  > coplanarity score (a Sampson distance from the 8 correspondences to the coplanarity stratum of the 8-point
  > reconstruction; not a conditioning score) had AUC 0.983 against 0.937 for the DEGENSAC inlier count (difference
  > 0.046, 95% CI [0.027, 0.069]; H1 supported). This result is for the pre-registered representation (unit columns,
  > largest-triangle base triple): the score is invariant to the choice of representatives and base triple only to
  > first order, and a unit test shows that the representatives change it beyond first order. The CAPRI conditioning
  > scores had
  > AUC 0.857 (capri_kappa; difference to DEGENSAC -0.080, 97.5% CI [-0.102, -0.054]) and 0.871 (capri_ls = L(S);
  > -0.065, 97.5% CI [-0.091, -0.037]). Against the continuous 2-D homography tests, capri_coplanarity was equivalent
  > within the +/-0.01 margin to degensac_e6 (difference 0.0026, 90% CI [-0.0036, 0.0098]) and to homography_ls6
  > (difference -0.0001, 90% CI [-0.0069, 0.0074]). N5 ("L(S) loses to DEGENSAC") stands. No real-image claim is made.

- The GOAL row-17 note should carry the same qualitative scope clause. They must not quote 0.915, 1.06, 0.982, 0.9747, 0.9686 or any other figure of the first
  bullet.
- After these edits the experiment was re-run unchanged; all five `degeneracy_matched_*.csv` files reproduce the stored ones
  (see the build report).

### Provenance of this pre-registration (2026-10-12, final review; text only, no re-run)

- This file, including the revision log of referee round 1 and the pre-run amendments P1-P14, was first committed in
  3653699 (2026-10-12 07:41 UTC), after the single confirmatory run (log finished about 03:26). That commit message
  ("Written and revised before confirmatory runs") and the header of this file are therefore not shown by git for
  the coplanarity component. 3653699 is not a pre-run timestamp for A and must not be cited as one.
- The pre-run status of Sections 1-10, the revision log and P1-P14 rests only on this document's own text and on
  scratch-file timestamps: design exploration 01:45-02:41; R2 chart check `chart_check.py`
  02:52; referee checks 02:48-02:56; implementation scripts from 03:11 (`t1.py`), P1 smoke script 03:21; the
  confirmatory log `confirmatory_run.log` 03:26. No earlier copy of this file was saved, so the wording at
  the time of the run cannot be shown.
- Exposure (restating P1): the P1 smoke runs printed pooled AUCs, CIs and verdicts on 36 confirmatory test samples
  (test trials 0-3, 3 samples per k) before the analysis code was final.
- The "Post-run record", the "Post-hoc checker note" and this entry were written after the run, as their titles say.
- Final-review fix (scratch-only numbers): the module docstring items (5)-(7), Section 2 and the row-17 sentence no
  longer state figures that only scratch scripts produced; the remaining scratch figures in this file are labelled
  "not reproducible from repository code". No analysis, threshold or result changed.
