/-
The clause of Proposition 5 on gauges: each measurement may be followed by per-point scalings and
an orthogonal map, and `z_t` is unchanged.

* `orth_norm`: an orthogonal matrix preserves Euclidean norms.
* `unitize_act`: the unit representatives of `l_k A a_k` are `sign(l_k) A unit(a_k)`.
* `grad_norm_act`: for orthogonal `A` and signs `s_k = ±1`, the gradient of `log|CR_t|` has the
  same Euclidean norm at `act A s Q` as at `Q` (chain rule; `log|CR_t|` is gauge-invariant).
* `pivotal_limit_gauge`: Proposition 5 with gauged measurements.
-/
import CapriFormal.PivotalLimit

open Matrix MeasureTheory ProbabilityTheory Filter Topology Metric

namespace Capri

variable {n : ℕ}

theorem dot_orth {A : Matrix (Fin 4) (Fin 4) ℝ} (hA : Aᵀ * A = 1) (x : Fin 4 → ℝ) :
    (A *ᵥ x) ⬝ᵥ (A *ᵥ x) = x ⬝ᵥ x := by
  rw [dotProduct_mulVec, ← mulVec_transpose, mulVec_mulVec, hA, one_mulVec, dotProduct_comm]

theorem euclid_sq (y : Fin 4 → ℝ) : ‖WithLp.toLp 2 y‖ ^ 2 = y ⬝ᵥ y := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  simp [dotProduct, Real.norm_eq_abs, sq_abs, sq]

theorem orth_norm {A : Matrix (Fin 4) (Fin 4) ℝ} (hA : Aᵀ * A = 1) (x : Fin 4 → ℝ) :
    ‖WithLp.toLp 2 (A *ᵥ x)‖ = ‖WithLp.toLp 2 x‖ := by
  have h : ‖WithLp.toLp 2 (A *ᵥ x)‖ ^ 2 = ‖WithLp.toLp 2 x‖ ^ 2 := by
    rw [euclid_sq, euclid_sq, dot_orth hA]
  exact (pow_left_inj₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).1 h

theorem orth_det {A : Matrix (Fin 4) (Fin 4) ℝ} (hA : Aᵀ * A = 1) : A.det ≠ 0 := by
  intro h
  have := congrArg Matrix.det hA
  rw [Matrix.det_mul, Matrix.det_transpose, h, mul_zero, Matrix.det_one] at this
  exact zero_ne_one this

/-- Unit representatives of a gauged configuration. -/
theorem unitize_act {A : Matrix (Fin 4) (Fin 4) ℝ} (hA : Aᵀ * A = 1) (l : Fin n → ℝ)
    (hl : ∀ k, l k ≠ 0) (Q : Config n) :
    unitize (act A l Q) = act A (fun k => SignType.sign (l k)) (unitize Q) := by
  funext k
  simp only [unitize, act]
  rw [WithLp.toLp_smul, norm_smul, orth_norm hA, Real.norm_eq_abs, mul_inv, smul_smul,
    Matrix.mulVec_smul, smul_smul]
  congr 1
  have hk := hl k
  rcases lt_or_gt_of_ne (hl k) with h | h
  · rw [abs_of_neg h, sign_neg h]; field_simp; simp [neg_div, div_eq_inv_mul]
  · rw [abs_of_pos h, sign_pos h]; field_simp; simp

/-- The gauge action as a continuous linear map. -/
noncomputable def actL (A : Matrix (Fin 4) (Fin 4) ℝ) (s : Fin n → ℝ) : Config n →L[ℝ] Config n :=
  LinearMap.toContinuousLinearMap
    { toFun := act A s
      map_add' := fun P Q => by funext k; simp [act, Matrix.mulVec_add, smul_add]
      map_smul' := fun c P => by
        funext k; simp [act, Matrix.mulVec_smul, smul_smul, mul_comm] }

theorem actL_apply (A : Matrix (Fin 4) (Fin 4) ℝ) (s : Fin n → ℝ) (Q : Config n) :
    actL A s Q = act A s Q := rfl

theorem actL_ec (A : Matrix (Fin 4) (Fin 4) ℝ) (s : Fin n → ℝ) (k : Fin n) (j : Fin 4) :
    actL A s (ec k j) = ∑ i : Fin 4, (s k * A i j) • ec k i := by
  funext m i'
  rw [actL_apply]
  simp only [act, ec, ev, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  by_cases hm : m = k
  · subst hm
    simp only [Pi.single_eq_same]
    rw [Finset.sum_eq_single i']
    · simp [Matrix.mulVec, dotProduct, Pi.single_apply]
    · intro b _ hb; simp [Pi.single_apply, Ne.symm hb]
    · simp
  · simp [Pi.single_eq_of_ne hm]

theorem Ut_act (t : Fin 6 → Fin n) {A : Matrix (Fin 4) (Fin 4) ℝ} (hA : A.det ≠ 0)
    {s : Fin n → ℝ} (hs : ∀ k, s k ≠ 0) {Q : Config n} (hQ : Q ∈ Ut t) : act A s Q ∈ Ut t := by
  have k : ∀ a b c d : Fin n, br (act A s Q a) (act A s Q b) (act A s Q c) (act A s Q d) =
      s a * s b * s c * s d * A.det * br (Q a) (Q b) (Q c) (Q d) := by
    intro a b c d; simp only [act]; rw [br_smul, br_mulVec]; ring
  obtain ⟨h1, h2, h3, h4⟩ := hQ
  refine ⟨?_, ?_, ?_, ?_⟩ <;> rw [k] <;>
    exact mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (hs _) (hs _))
      (hs _)) (hs _)) hA) ‹_›

/-- **Gradient norms are gauge-invariant** for orthogonal maps and sign changes. -/
theorem grad_norm_act (t : Fin 6 → Fin n) {A : Matrix (Fin 4) (Fin 4) ℝ} (hA : Aᵀ * A = 1)
    {s : Fin n → ℝ} (hs : ∀ k, s k = 1 ∨ s k = -1) {Q : Config n} (hQ : Q ∈ Ut t) :
    ‖grad (phi t) (act A s Q)‖ = ‖grad (phi t) Q‖ := by
  have hdet := orth_det hA
  have hs0 : ∀ k, s k ≠ 0 := fun k => by rcases hs k with h | h <;> rw [h] <;> norm_num
  have hs2 : ∀ k, s k ^ 2 = 1 := fun k => by rcases hs k with h | h <;> rw [h] <;> norm_num
  have hinv : phi t ∘ actL A s = phi t := by
    funext Q'; simp [phi, actL_apply, crAt_act' t A hdet s hs0]
  have hdiffQ' : DifferentiableAt ℝ (phi t) (act A s Q) :=
    ((logcr_contDiffOn t).differentiableOn (by norm_num)).differentiableAt
      ((Ut_isOpen t).mem_nhds (Ut_act t hdet hs0 hQ))
  have hchain : fderiv ℝ (phi t) Q = (fderiv ℝ (phi t) (act A s Q)).comp (actL A s) := by
    conv_lhs => rw [← hinv]
    rw [fderiv_comp Q (by rw [actL_apply]; exact hdiffQ') (actL A s).differentiableAt,
      (actL A s).fderiv, actL_apply]
  set y : Fin n → Fin 4 → ℝ := fun k i => fderiv ℝ (phi t) (act A s Q) (ec k i)
  have hpart : ∀ k j, fderiv ℝ (phi t) Q (ec k j) = s k * ∑ i, A i j * y k i := by
    intro k j
    rw [hchain, ContinuousLinearMap.comp_apply, actL_ec, map_sum]
    simp only [map_smul, smul_eq_mul, y, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => by ring
  have hAt : (Aᵀ)ᵀ * Aᵀ = 1 := by rw [transpose_transpose]; exact mul_eq_one_comm.1 hA
  have hblock : ∀ k, ∑ j, (∑ i, A i j * y k i) ^ 2 = ∑ i, y k i ^ 2 := by
    intro k
    have h := dot_orth hAt (y k)
    simp only [dotProduct, Matrix.mulVec, transpose_apply] at h
    simpa [sq] using h
  have hsq : ‖grad (phi t) Q‖ ^ 2 = ‖grad (phi t) (act A s Q)‖ ^ 2 := by
    rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity),
      Real.sq_sqrt (by positivity)]
    change ∑ kj : Fin n × Fin 4, ‖fderiv ℝ (phi t) Q (ec kj.1 kj.2)‖ ^ 2 =
      ∑ kj : Fin n × Fin 4, ‖y kj.1 kj.2‖ ^ 2
    simp only [Real.norm_eq_abs, sq_abs, hpart, Fintype.sum_prod_type, mul_pow, hs2, one_mul,
      hblock]
  exact ((pow_left_inj₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).1 hsq).symm

/-- The standardised discrepancy computed from gauged measurements `A₁ l₁ X` and `A₂ l₂ Y`
(gradients at their unit representatives). -/
noncomputable def zgauge (t : Fin 6 → Fin n) (μ r : ℝ) (A₁ A₂ : Matrix (Fin 4) (Fin 4) ℝ)
    (l₁ l₂ : Fin n → ℝ) (X Y : Config n) : ℝ :=
  |phi t (act A₁ l₁ X) - phi t (act A₂ l₂ Y)| /
    (μ * r * (‖grad (phi t) (unitize (act A₁ l₁ X))‖ + ‖grad (phi t) (unitize (act A₂ l₂ Y))‖))

theorem sign_pm {x : ℝ} (hx : x ≠ 0) : (SignType.sign x : ℝ) = 1 ∨ (SignType.sign x : ℝ) = -1 := by
  rcases lt_or_gt_of_ne hx with h | h
  · right; simp [sign_neg h]
  · left; simp [sign_pos h]

/-- Gauges with orthogonal maps and nonzero scalings do not change `z_t`. -/
theorem zgauge_eq (t : Fin 6 → Fin n) (μ r : ℝ) {A₁ A₂ : Matrix (Fin 4) (Fin 4) ℝ}
    (hA₁ : A₁ᵀ * A₁ = 1) (hA₂ : A₂ᵀ * A₂ = 1) {l₁ l₂ : Fin n → ℝ} (hl₁ : ∀ k, l₁ k ≠ 0)
    (hl₂ : ∀ k, l₂ k ≠ 0) (q u v : Config n) (hu : unitize (q + r • u) ∈ Ut t)
    (hv : unitize (q + r • v) ∈ Ut t) :
    zgauge t μ r A₁ A₂ l₁ l₂ (q + r • u) (q + r • v) = zstat t μ r q u v := by
  simp only [zgauge, zstat, phi]
  rw [crAt_act' t A₁ (orth_det hA₁) l₁ hl₁, crAt_act' t A₂ (orth_det hA₂) l₂ hl₂,
    unitize_act hA₁ l₁ hl₁, unitize_act hA₂ l₂ hl₂,
    grad_norm_act t hA₁ (fun k => sign_pm (hl₁ k)) hu,
    grad_norm_act t hA₂ (fun k => sign_pm (hl₂ k)) hv]

/-- **Proposition 5 with gauges.** Each measurement may be followed by per-point scalings and an
orthogonal map (any choice, even depending on `q`, `r` and the noise); the standardised
discrepancy still converges in distribution to `|N(0,1)|/(√2 μ)`, uniformly in `q ∈ K`. -/
theorem pivotal_limit_gauge (t : Fin 6 → Fin n) {K : Set (Config n)} (hK : IsCompact K)
    (hKU : K ⊆ Ut t) (hKunit : ∀ q ∈ K, ∀ k, ‖WithLp.toLp 2 (q k)‖ = 1)
    {γ μ : ℝ} (hγ : 0 < γ) (hgrad : ∀ q ∈ K, γ ≤ ‖grad (phi t) q‖) (hμ : 0 < μ)
    (A₁ A₂ : Config n → ℝ → Coord n × Coord n → Matrix (Fin 4) (Fin 4) ℝ)
    (l₁ l₂ : Config n → ℝ → Coord n × Coord n → Fin n → ℝ)
    (hA₁ : ∀ q r ω, (A₁ q r ω)ᵀ * A₁ q r ω = 1) (hA₂ : ∀ q r ω, (A₂ q r ω)ᵀ * A₂ q r ω = 1)
    (hl₁ : ∀ q r ω k, l₁ q r ω k ≠ 0) (hl₂ : ∀ q r ω k, l₂ q r ω k ≠ 0) :
    ∀ ε > 0, ∃ r₁ > 0, ∀ r, 0 < r → r ≤ r₁ → ∀ q ∈ K, ∀ x : ℝ,
      |((stdGaussian (Coord n)).prod (stdGaussian (Coord n))).real
          {ω | zgauge t μ r (A₁ q r ω) (A₂ q r ω) (l₁ q r ω) (l₂ q r ω)
            (q + r • toCfg ω.1) (q + r • toCfg ω.2) ≤ x}
        - (gaussianReal 0 1).real {z | |z| / (Real.sqrt 2 * μ) ≤ x}| ≤ ε := by
  obtain ⟨δ, hδ, hsub⟩ := hK.exists_cthickening_subset_open (Ut_isOpen t) hKU
  refine pivotal_limit_of t hgrad hγ hμ (fun q r ω => zgauge t μ r (A₁ q r ω) (A₂ q r ω)
    (l₁ q r ω) (l₂ q r ω) (q + r • toCfg ω.1) (q + r • toCfg ω.2)) ?_
  intro β hβ
  obtain ⟨C, r₀, -, hr₀, h⟩ := pivotal_taylor t hK hKU hKunit hγ hgrad hμ hβ
  refine ⟨C, min r₀ (δ / (4 * β + 1)), lt_min hr₀ (by positivity), ?_⟩
  intro q hq ω h1 h2 r hr hrr
  have hin : ∀ w : Config n, ‖w‖ ≤ β → unitize (q + r • w) ∈ Ut t := by
    intro w hw
    refine hsub (mem_cthickening_of_dist_le _ q δ K hq ?_)
    rw [dist_eq_norm]
    refine le_trans (unitize_dist q (r • w) (hKunit q hq)) ?_
    have hr2 : r ≤ δ / (4 * β + 1) := le_trans hrr (min_le_right _ _)
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos hr]
    rw [le_div_iff₀ (by positivity)] at hr2
    nlinarith
  show |zgauge t μ r _ _ _ _ _ _ - _| ≤ C * r
  rw [zgauge_eq t μ r (hA₁ q r ω) (hA₂ q r ω) (hl₁ q r ω) (hl₂ q r ω) q _ _
    (hin _ h1) (hin _ h2)]
  exact h q hq _ _ h1 h2 r hr (le_trans hrr (min_le_left _ _))

end Capri
