# GOAL.md: accuracy targets and their status

Each target names the test or experiment that checks it. "Met" means the check runs in
this repository and passes. Numbers come from the CSVs in `experiments/results/`, never
from this file. The checks are simulations and exact computations; real-data checks are listed
in `DATA.md`.

## 3D-reconstruction reliability layer

| Goal | Target | Check | Status |
|---|---|---|---|
| Bracket accuracy | Brackets accurate to ~1e-12 relative for well-conditioned 4x4 matrices; exact integer brackets available for verification | `tests/test_brackets.py`, `forensics/geometry/brackets.py::exact_integer_determinant` | Met |
| Invariance | J1-J3 invariant under random PGL(4) maps (relative error < 1e-8) | `run_invariance.py`, `tests/test_invariants.py` | Met (median ~1e-15) |
| Plücker relations | All 15 Grassmann-Plücker relations hold identically; Sigma_1-3 do not generate them (vector b*) | `experiments/symbolic/verify_plucker.py`, `tests/test_plucker.py` | Met |
| Stratum codimension | Degeneracy strata have codimension m - 3 | `experiments/symbolic/verify_plucker.py` (ideal of 4x4 minors) | Met |
| Drift bound | Drift admission uses a proven bound | `compute_drift_threshold` = tau_exact (Theorem 6.1); `run_drift_admission.py` asserts 100% admission of bounded drift | Met. The bound is valid but rejects no excessive drift, so the consistency test is the pencil statistic (the drift-test target). |
| Conditioning threshold | Conditioning threshold justified by data | `run_ablation.py` (calibrated operating point); `pipeline.main_pipeline.calibrated_conditioning_threshold` | Met. A fixed threshold L < 1000 accepts < 1% of generic tuples and is reported only as a comparison point. |
| Conditioning frame | Conditioning is computed in a well-defined frame | `algorithms.utils.whiten_homogeneous_points` (Tyler frame), `tests/test_pipeline_e2e.py` | Met. kappa_J and L(S) are not PGL(4)-invariant, so they are computed in a canonical frame. |
| Stratum classification | Stratum classification recovers compound configurations | `run_stratification.py` (rank classifier vs single threshold) | Met by the rank classifier. A single threshold has 0 compound recall. |
| Use of rho | rho(S) used only where it can detect something | syzygy-residual experiment and ablation study use the independent-estimation model; rho = 0 on one point set (Corollary 2.6) | Met |
| Confidence fusion | Confidence fusion beats the majority baseline | `run_fusion_metrics.py` (neural fusion), `run_logistic_fusion.py` (logistic fusion, `docs/prereg/logistic_fusion.md`) | Neural fusion: **not met** (predicts the majority class, MCC 0; off by default). Logistic fusion: met only with both measurements in a common, known chart (MCC 0.774 vs 0.326 for floor + majority; +0.058 [0.047, 0.071] over the best single input); with deployable inputs no gain over the pencil input was shown (+0.006 [-0.005, 0.017]). Off by default. |
| Real-image benefit | Real-image benefit of CAPRI filtering | `run_real_world_baselines.py` (`docs/prereg/real_reconstruction.md`) | **Not met (no difference).** On the 23 eligible IMC 2025 scenes the layer leaves mAA@10 unchanged (difference -0.0027, 95% CI [-0.0082, 0.0020]; equivalent within 0.01). |
| Real-scene size of the consistency test | Size in [3.5%, 6.5%] on held-out real scenes | `run_real_pencil.py` (`docs/prereg/real_pencil.md`) | **Not met** by any of the five statistics (1.7-47.4% on 534 test windows from 10 scenes); no power reported. |
| Backend agreement | Python and C++ backends agree | `run_runtime.py`, `run_pencil_runtime.py` (`Outputs_Agree`) | Met |
| Drift test | Drift test that rejects excessive drift while admitting bounded drift | `run_pencil_drift.py`, `docs/MATHEMATICS.md` Section 4 | Met in simulation (AUC 0.984, TPR 0.927 at 5% FPR; isotropic chart noise). With triangulated pixel noise the test does better (AUC 0.971 vs 0.939, `noise_model_summary.csv`). On injected point-level inconsistency a known-sigma GLRT or the Sampson error matches or beats it (`detector_benchmark_cells.csv`). |
| Conditioning score | Conditioning score that predicts noise error | `run_kappa_conditioning.py`, Proposition 5.1 | Met: kappa_J Spearman 0.995 (L(S) 0.844) |
| Checked mathematics | Every mathematical claim has a proof or a labelled status and a check that can fail | `docs/MATHEMATICS.md`, `experiments/symbolic/verify_pencil.py`, `tests/test_math_lock.py` | Met |
| Tuple selection | Tuple selection with a proven approximation guarantee | `run_greedy_efficiency.py` (MaxCov columns) | Met: 0 of 200 runs below 1 - 1/e (greedy with an L(S) penalty: 44 of 200) |
| Degeneracy test | CAPRI beats a standard degeneracy test | `run_synthetic_baselines.py`, `run_degeneracy_matched.py` (`docs/prereg/degeneracy_matched.md`), representation-robustness check (`coplanarity_robustness_comparisons.csv`) | **Not met.** kappa_J (AUC 0.927) and L(S) (0.923) lose to DEGENSAC (0.961). A coplanarity score with matched inputs reached 0.983 vs 0.937 for the DEGENSAC inlier count in its pre-registered representation, but with affine representatives X/X_4 no difference was shown (-0.021, 99% CI [-0.044, 0.011]). |
| Trust calibration | Trust score gives calibrated probabilities | `run_trust_calibrated.py` (`docs/prereg/trust_calibrated.md`) | Partly met: marginally calibrated (ECE 0.018 [0.010, 0.028]) but weak (AUC 0.626) and not calibrated within strata (worst ECE 0.349) or under shift (0.324). Not used in the pipeline (point coverage is reported instead). |

## Forensic extension

| Goal | Target | Check | Status |
|---|---|---|---|
| Synthetic null | Structural z-scores ~ N(0,1) on consistent synthetic segments | `synthetic_null.py` (KS distance, tail fraction) | Met in simulation |
| Multi-view fixture | Noise-free multi-view pencil statistic ~ 0; monotone response to injected non-rigidity | `multiview_synthetic.py`, `tests/test_forensics/` | Met in simulation (noise-free Q0.1 <= 5.1e-10; noise slope 0.997) |
| Video fixture | Video path runs end to end and responds to time-varying warps | `video_fixture.py` | Met on rendered clips |
| Real-media null | Held-out authentic rejection rate in [3.5%, 6.5%] at alpha = 5% | `real_media_null.py` | Needs real data |
| Held-out detection | Held-out-generator AUC >= 0.70, CI excluding 0.5, metadata AUC <= 0.55 | `heldout_detection.py` | Needs real data. Stop and report if they fail. |
| Compression robustness | Robust to JPEG and resizing (Delta-FPR, flip rate) | `compression_sweep.py` | Needs real data |

## Engineering

- Every result file carries a provenance header (seed, backend, commit, versions).
- No result is typed into a script. No random draw stands in for a measurement.
- `make experiments && make figures` regenerates every simulation result and figure.
