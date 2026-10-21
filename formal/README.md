# Lean formalization (CAPRI)

Lean 4 v4.35.0 with Mathlib v4.35.0 (`lakefile.toml`, `lean-toolchain`). Check with
`bash formal/check.sh`: `lake build` must succeed with no `sorry`, and twenty-three controls, each a
deliberately false version of a statement, must be rejected. Recorded run: `check_output.txt`.
(Mathlib's binary cache could not be downloaded in the cloud container; it was built from source.)

What is formalized, and what is not. Statement numbers refer to `docs/MATHEMATICS.md`.

| Lean theorem | File | Statement | Scope |
|---|---|---|---|
| `br_mulVec`, `br_smul` | Pencil | brackets scale by det A and by the point scales | full |
| `cr_gauge` | Pencil | Lemma 2.1: pencil cross-ratios (sign included) are gauge-invariant | full |
| `frame_cr` | Pencil | Prop 3.5: CR_12(3,4;5,k) = z/w, CR_13(2,4;5,k) = y/w, CR_23(1,4;5,k) = x/w | full (Lean's t/0 = 0 makes both sides 0 when w = 0) |
| `frame_brackets` | Pencil | [234k] = -x, [134k] = y, [124k] = -z, [123k] = w | full |
| `plucker`, `cr_add_cr` | Pencil | Thm 2.5: GP_ab = 0, and CR_ab(c,d;e,f) + CR_ab(c,e;d,f) = 1 | full |
| `stabiliser` | Rank | Lemma 3.6, linear-algebra core: X p_i = -mu_i p_i on a basis and on q = sum c_i p_i with all c_i != 0 force mu_i = nu and X = -nu I | full with `frame_coeffs`, `orbitMap_ker` below |
| `frame_block_rank` | Rank | Thm 3.7, lower-bound block: the three frame gradients are linearly independent when x, y, z != 0 | full; the theorem itself is `rank_pencil_map` below |
| `Un`, `crAt`, `Phi`, `act`, `Phi_act`, `Phi_differentiableAt` | RankFull | U_n; the pencil map Φ(P) = (log abs CR_t(P))_t over a family of pencils (ordered 6-tuples of distinct points); Φ is gauge-invariant and differentiable on U_n | full |
| `fderiv_orbit` | RankFull | Thm 3.7, upper bound: DΦ(P) vanishes on (X, μ) ↦ (X p_k + μ_k p_k)_k (curve through the gauge orbit, chain rule) | full |
| `br_cramer`, `frame_coeffs`, `Un_ne_zero`, `orbitMap_ker`, `orbitMap_range_finrank` | RankFull | Lemma 3.6 with Cramer's rule: p_4 = sum c_i p_i with every c_i ≠ 0; the orbit differential has kernel spanned by (I, -1) and image of dimension n + 15 | full |
| `rank_le` | RankFull | Thm 3.7: rank DΦ(P) ≤ 3n - 15 | full |
| `cr_lin`, `cr_scale`, `frame_normalise` | RankFull | Prop 3.5 transported by the gauge: the frame pencils through p_0..p_4 and q are z/w, y/w, x/w of A q, with nonzero coordinates on U_n | full |
| `hasDerivAt_log_ratio`, `fderiv_line`, `framePencil`, `rank_ge` | RankFull | Thm 3.7: rank DΦ(P) ≥ 3(n - 5) for any family containing the frame pencils of every point k ≥ 5 | full |
| `rank_pencil_map`, `rank_all_pencils` | RankFull | **Thm 3.7**: rank DΦ(P) = 3n - 15 on U_n for every n ≥ 5, for any such family and for all pencils | full |
| `log_abs_change`, `neg_log_mono`, `logJ_bound` | LogBound | Thm 5.3 from the bracket perturbation on | full with `logJ_bound_vectors` below |
| `hadamard`, `det_perturb` | Hadamard | Hadamard's inequality; if every ‖q_i‖ ≤ 1 and ‖u_i‖ ≤ δ, the determinant moves by at most (1 + δ)^n - 1 (multilinear expansion over subsets of replaced columns) | full |
| `br_eq_det`, `br_hadamard`, `br_perturb` | Hadamard | the same for the bracket of four vectors of ℝ⁴ (Euclidean norms) | full |
| `three_products`, `tau_exact` | Hadamard | Thm 6.1: each Plücker relation, with each bracket from its own representatives within δ of unit vectors, is at most 3ε(2 + ε), ε = (1 + δ)^4 - 1 | full (Σ_1-Σ_3 are such relations up to sign) |
| `logJ_bound_vectors` | Hadamard | Thm 5.3 from the vectors on | full |
| `many_above`, `exists_above`, `size_bound` | Quantile | Prop 4.5: union bound and Markov bound for "fewer than k values are <= c" (the k-th order statistic exceeds c), any joint law | full with `size_bound_interp` below |
| `interp`, `interp_le`, `numpy_indices`, `card_le_of_interp_gt`, `size_bound_interp` | QuantileInterp | Prop 4.5 for numpy's linearly interpolated p-quantile, k = ⌈(T-1)p⌉ + 1 | full |
| `sub_stdGaussian`, `stdGaussian_map_inner_div`, `limit_law` | LimitLaw | Prop 4.4, the limit law: u~ - v~ ~ √2 N(0, I); for g ≠ 0, abs⟨g, w⟩ / (2 μ ‖g‖) has the law of abs N(0,1) / (√2 μ) whatever g | full |
| `cdf_coupling`, `uniform_in_prob`, `half_normal_modulus`, `limit_theorem_uniform` | LimitLaw | Prop 4.4, the convergence: coupling inequality, uniform convergence in probability, and sup_q sup_x of the CDF error → 0 | full with the rows below |
| `taylor_uniform`, `fderiv_lip_uniform`, `Ut`, `logcr_contDiffOn` | Taylor | uniform second-order Taylor bound and Lipschitz derivative on compact sets; log abs CR_t is C² where its brackets are nonzero | full |
| `scale_family` | Taylor | **Thm 4.1**, one pencil: abs(D_t - r abs(Dφ_t(q)(u - v))) ≤ C r² uniformly on compact sets, for any gauges | full |
| `ostat`, `ostat_lip`, `ostat_smul`, `qp`, `qlip`, `qsmul` | ScaleFamily | Lemma 4.0: order statistics and numpy's interpolated quantile are 1-Lipschitz in the sup norm and positively homogeneous | full |
| `scale_family_S`, `size_power_limit` | ScaleFamily | **Thm 4.1** for S = Q_p(D)/δ: abs(S - (r/δ) Z) ≤ C r²/δ; P(S_δ > c) → P(Z > c/f) as δ → 0 | full; the power needs Z to have no atom at c/f (the text said only "at c"; corrected) |
| `unitize`, `zstat`, `Lstat`, `grad_sub_le`, `unit_dist`, `unitize_dist`, `pivotal_taylor` | Pivotal | **Prop 4.4**, the Taylor step: abs(z_t - L_t) ≤ C r on bounded noise, gradients at the measured unit representatives | full |
| `toCfg`, `apply_toCfg`, `pivotal_limit_of`, `pivotal_limit` | PivotalLimit | **Prop 4.4**: with ũ, ṽ independent standard Gaussian in the 4n coordinates, z_t → abs N(0,1)/(√2 μ) in distribution, uniformly in q | full |
| `orth_norm`, `unitize_act`, `grad_norm_act`, `zgauge_eq`, `pivotal_limit_gauge` | PivotalGauge | **Prop 4.4**, gauge clause: per-point scalings and orthogonal maps leave z_t unchanged | full |
| `integral_sqrt_one_sub_sq_sym`, `small_ball_lower_integral` | SmallBall | Prop 5.4: the lower bound (2/π)(d√(1-d²) + arcsin d) is the integral of the density of fact (i) | full (the law itself is `law_abs_inner` below) |
| `withDensity_Ioo_le`, `mean_inv_vol2`, `mean_inv_dist`, `small_ball_constant` | SmallBall | Prop 5.4: the conditional density step, E[1/vol_2] = 4/π, E[1/dist] = 2 and the constant 32/π² as integrals | full; combined in `SphereSmallBall` below |
| `unifS3`, `unifS3_cone`, `unifS3_rot`, `rot_to_e0`, `rot_to_e01` | Sphere | the uniform probability measure on S³ (normalised surface measure); for a scale-invariant event it is the volume of the cone in the unit ball over π²/2; rotation invariance | full |
| `unif_abs_inner_lt`, `law_abs_inner` | SphereLaw | Prop 5.4, fact (i): P(abs ⟨n, x⟩ < a) = Ψ(a) = (2/π)(a√(1-a²) + arcsin a) for a unit n and 0 < a ≤ 1; abs ⟨n, x⟩ has density (4/π)√(1-t²) on (0, 1) | full (proved from cone volumes in ℝ⁴ = ℝ¹ × ℝ³, not from the Gamma/Beta route of the text) |
| `unif_distSq_lt`, `law_distSq` | SphereLaw | Prop 5.4, fact (ii): for an orthonormal pair u₁, u₂, 1 - ⟨u₁, x⟩² - ⟨u₂, x⟩² is uniform on (0, 1) | full (cone volumes in ℝ² × ℝ²); fact (iii) is not used by the proof and is not formalized |
| `Nv`, `normNv_sq`, `normNv_sq_unit`, `inner_prob`, `inner_le`, `inner_ge`, `lint_Nv_x3`, `lint_invN` | SphereSmallBall | Prop 5.4, the conditioning: det M = ⟨N, x₄⟩, ‖N‖² is the Gram determinant, P(abs det M < d given x₁, x₂, x₃) = Ψ(min(d/‖N‖, 1)), E[1/‖N‖] = 8/π | full |
| `small_ball_upper`, `small_ball_lower`, `small_ball_limit` | SphereSmallBall | **Prop 5.4**: for four independent uniform columns (product measure `unif4`), (2/π)(d√(1-d²) + arcsin d) ≤ P(abs det M < d) (0 < d ≤ 1), P(abs det M < d) ≤ (32/π²) d (d > 0), and P(abs det M < d)/d → 32/π² as d → 0⁺ | full |
| `map_four`, `bracket_small_ball`, `kappa_tail_six`, `L_tail_six` | SphereSmallBall | **Prop 5.4**, consequences: for six independent uniform points of S³ (any probability space), every bracket has the small-ball bound; P(κ_J > M) ≤ 2816/(π² M); P(L(S) > l) ≤ 15 (32/π²) √(8/l), and P(L(S) > l) ≥ (2/π) √(8/l) for l ≥ 8 | full (L(S) = 8/δ_min² as in the text) |
| `kappa_tail`, `L_tail` | SmallBall | Prop 5.4: P(κ_J > M) ≤ 11·C·8/M (2816/(π² M) with C = 32/π²) and P(abs b_1 < √(8/l)) ≤ P(L(S) > l) ≤ 15·C·√(8/l), from a per-bracket small-ball bound | full given the small-ball bound (κ_J ≤ 8/δ_J is `kappa_le`) |
| `ec`, `grad`, `brf`, `brPart`, `partial_logbr`, `cofactor_sq_le`, `grad_logbr_le` | Kappa | Lemma 3.1 for brackets (the gradient of log abs [abcd] is the cofactor vector over the bracket); ‖∇ log abs [abcd]‖ ≤ 2/abs [abcd] at unit representatives | full |
| `Jq`, `logJ`, `kappaJ`, `deltaJ`, `deltaMin`, `grad_logJ`, `kappa_le`, `kappa_le_L`, `no_reverse` | Kappa | **Prop 5.2**: κ_J ≤ 8/δ_J ≤ 8/δ_min = 2√(2L); a unit configuration with [1236] = 0 and all 11 used brackets nonzero | full |
| `gaussian_abs_lt`, `gaussian_abs_gt`, `noise_law` | NoiseLaw | **Prop 5.1**: the median of M = max_k abs(σ⟨g_k, X⟩) lies between Φ⁻¹(3/4)σκ and Φ⁻¹(11/12)σκ | full (the decimals 0.674, 1.383 are not part of the statement) |
| `used_card`, `unused_eq`, `axis_points` | Strata | Prop 5.5: J1-J3 use 11 brackets, the unused four are [1236], [1245], [1456], [2356]; each J_k has two axis points in all four brackets and four points in exactly two | full |
| `inv_entry`, `rate_upper`, `rate_lower` | Strata | Prop 5.5: entries of M⁻¹ are cofactors over det M; the inequalities giving κ_J ≤ 8/(c'd) and the lower rate c/d | full with `Rate` below |
| `rate_used`, `rate_unused`, `normalise`, `normalise_curve` | Rate | **Prop 5.5**: c/d ≤ κ_J(P_d) ≤ C/d near a used zero bracket; κ_J bounded near an unused one; the text's P_d is such a curve | full |
| `orbit`, `separation_signed`, `Un_isOpen`, `separation_local` | Separation | **Cor 3.8**: (b) signed frame cross-ratios agree iff Q ∈ G·P; (a) near P, equal frame values log abs CR imply Q ∈ G·P | full for (b); (a) proves orbit membership (that the gauge can be taken near the identity is not formalized) |

Not formalized: fact (iii) of Prop 5.4 (used only by a numerical check), Lemma 2.2, Props 2.3-2.4,
Cor 2.6, Thm 3.2 and Cor 3.3 (superseded by Thm 3.7 and Cor 3.8), Prop 4.3, Props 6.2-6.3,
Prop 7.1, Section 10, Remarks 3.9, 4.2 and 5.6. Those keep their written proofs and computer
checks.
