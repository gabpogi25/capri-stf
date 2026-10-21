/-
Proposition 9 (MATHEMATICS.md Proposition 5.4): small ball for four i.i.d. uniform points of S³.

For `M = [x₁ x₂ x₃ x₄]` with independent uniform columns on S³:
* `small_ball_upper`: `P(|det M| < d) ≤ (32/π²) d` for `d > 0`;
* `small_ball_lower`: `(2/π)(d √(1 - d²) + arcsin d) ≤ P(|det M| < d)` for `0 < d ≤ 1`;
* `small_ball_limit`: `P(|det M| < d)/d → 32/π²` as `d → 0⁺`;
* `kappa_tail_six`, `L_tail_six`: for six i.i.d. uniform points,
  `P(κ_J > M) ≤ 2816/(π² M)`, and `P(L(S) > l)` lies between `(2/π) √(8/l)` (for `l ≥ 8`)
  and `15 (32/π²) √(8/l)`, so `P(L(S) > l) = Θ(l^{-1/2})`.

The proof follows the written proof in docs/MATHEMATICS.md. `det M = ⟨N, x₄⟩` with `N` the cofactor vector of `x₁, x₂, x₃`, so
given the first three columns `P(|det M| < d) = Ψ(min(d/‖N‖, 1)) ≤ (4/π) d/‖N‖` (fact (i)).
`‖N‖² = (1 - t²) · D` with `t = ⟨x₁, x₂⟩` and `D` the squared distance of `x₃` to the plane of
`x₁, x₂`, so `E[1/‖N‖] = E[1/√(1 - t²)] E[1/√D] = (4/π) · 2` (facts (i) and (ii), Fubini).
The four columns are modelled by the product measure `unif4`.
-/
import CapriFormal.SphereLaw
import CapriFormal.SmallBall
import CapriFormal.Kappa
import Mathlib.Probability.Independence.Basic

open MeasureTheory Metric Set Real Filter Topology
open scoped RealInnerProductSpace ENNReal

namespace Capri

/-! ### The cofactor vector -/

/-- The cofactor vector `N(x₁, x₂, x₃)`, with `⟨N, y⟩ = [x₁ x₂ x₃ y]`. -/
noncomputable def Nv (x₁ x₂ x₃ : E4) : E4 :=
  WithLp.toLp 2 fun j => br (WithLp.ofLp x₁) (WithLp.ofLp x₂) (WithLp.ofLp x₃) (ev j)

theorem inner_Nv (x₁ x₂ x₃ y : E4) :
    ⟪Nv x₁ x₂ x₃, y⟫ = br (WithLp.ofLp x₁) (WithLp.ofLp x₂) (WithLp.ofLp x₃) (WithLp.ofLp y) := by
  simp only [Nv, PiLp.inner_apply, Fin.sum_univ_four, br_expand, ev]
  simp
  ring

/-- `‖N‖²` is the Gram determinant of `x₁, x₂, x₃`. -/
theorem normNv_sq (x₁ x₂ x₃ : E4) :
    ‖Nv x₁ x₂ x₃‖ ^ 2 = ⟪x₁, x₁⟫ * ⟪x₂, x₂⟫ * ⟪x₃, x₃⟫ + 2 * ⟪x₁, x₂⟫ * ⟪x₁, x₃⟫ * ⟪x₂, x₃⟫
      - ⟪x₁, x₁⟫ * ⟪x₂, x₃⟫ ^ 2 - ⟪x₂, x₂⟫ * ⟪x₁, x₃⟫ ^ 2 - ⟪x₃, x₃⟫ * ⟪x₁, x₂⟫ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_Nv]
  simp only [Nv, PiLp.inner_apply, Fin.sum_univ_four, br_expand, ev]
  simp
  ring

/-- For unit vectors, `‖N‖² = (1 - a²)(1 - b²) - (c - ab)²` with `a = ⟨x₁, x₂⟩`, `b = ⟨x₁, x₃⟩`,
`c = ⟨x₂, x₃⟩`. -/
theorem normNv_sq_unit {x₁ x₂ x₃ : E4} (h₁ : ‖x₁‖ = 1) (h₂ : ‖x₂‖ = 1) (h₃ : ‖x₃‖ = 1) :
    ‖Nv x₁ x₂ x₃‖ ^ 2 = (1 - ⟪x₁, x₂⟫ ^ 2) * (1 - ⟪x₁, x₃⟫ ^ 2) - (⟪x₂, x₃⟫ - ⟪x₁, x₂⟫ * ⟪x₁, x₃⟫) ^ 2 := by
  have e : ∀ x : E4, ‖x‖ = 1 → ⟪x, x⟫ = 1 := fun x hx => by
    rw [real_inner_self_eq_norm_sq, hx, one_pow]
  rw [normNv_sq, e x₁ h₁, e x₂ h₂, e x₃ h₃]; ring

theorem normNv_le_one {x₁ x₂ x₃ : E4} (h₁ : ‖x₁‖ = 1) (h₂ : ‖x₂‖ = 1) (h₃ : ‖x₃‖ = 1) :
    ‖Nv x₁ x₂ x₃‖ ≤ 1 := by
  have ha := abs_real_inner_le_norm x₁ x₂
  have hb := abs_real_inner_le_norm x₁ x₃
  rw [h₁, h₂, one_mul] at ha; rw [h₁, h₃, one_mul] at hb
  have ha2 : ⟪x₁, x₂⟫ ^ 2 ≤ 1 := by rw [← sq_abs]; nlinarith [abs_nonneg ⟪x₁, x₂⟫]
  have hb2 : ⟪x₁, x₃⟫ ^ 2 ≤ 1 := by rw [← sq_abs]; nlinarith [abs_nonneg ⟪x₁, x₃⟫]
  have h := normNv_sq_unit h₁ h₂ h₃
  have hsq : ‖Nv x₁ x₂ x₃‖ ^ 2 ≤ 1 := by
    rw [h]; nlinarith [sq_nonneg (⟪x₂, x₃⟫ - ⟪x₁, x₂⟫ * ⟪x₁, x₃⟫), sq_nonneg ⟪x₁, x₂⟫,
      sq_nonneg ⟪x₁, x₃⟫, mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hb2)]
  nlinarith [norm_nonneg (Nv x₁ x₂ x₃)]

/-! ### Bounds on Ψ -/

theorem Psi_monoOn : MonotoneOn Psi (Icc 0 1) := by
  refine monotoneOn_of_deriv_nonneg (convex_Icc 0 1) Psi_continuous.continuousOn
    (fun y hy => ?_) (fun y hy => ?_) <;> rw [interior_Icc] at hy
  · exact (hasDerivAt_Psi (by linarith [hy.1]) hy.2).differentiableAt.differentiableWithinAt
  · rw [(hasDerivAt_Psi (by linarith [hy.1]) hy.2).deriv]
    have := Real.pi_pos; positivity

/-- `Ψ(a) ≤ (4/π) a` on `[0, 1]`: the density of fact (i) is at most `4/π`. -/
theorem Psi_le {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) : Psi a ≤ 4 / π * a := by
  have hpi := Real.pi_pos
  have hd : ∀ y ∈ Ioo (0 : ℝ) 1, HasDerivAt (fun y => 4 / π * y - Psi y)
      (4 / π * 1 - 4 / π * √(1 - y ^ 2)) y := fun y hy =>
    ((hasDerivAt_id y).const_mul (4 / π)).sub (hasDerivAt_Psi (by linarith [hy.1]) hy.2)
  have hm : MonotoneOn (fun y => 4 / π * y - Psi y) (Icc 0 1) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 1)
      ((continuous_const.mul continuous_id).sub Psi_continuous).continuousOn
      (fun y hy => ?_) (fun y hy => ?_) <;> rw [interior_Icc] at hy
    · exact (hd y hy).differentiableAt.differentiableWithinAt
    · rw [(hd y hy).deriv]
      have : √(1 - y ^ 2) ≤ 1 := Real.sqrt_le_one.2 (by nlinarith)
      have : 0 ≤ 4 / π := by positivity
      nlinarith
  have := hm ⟨le_rfl, zero_le_one⟩ ⟨ha0, ha1⟩ ha0
  simp only [mul_zero, Psi_zero, sub_zero] at this
  linarith

theorem le_arcsin_self {x : ℝ} (hx0 : 0 ≤ x) (hx1 : x ≤ 1) : x ≤ arcsin x := by
  have hd : ∀ y ∈ Ioo (0 : ℝ) 1, HasDerivAt (fun y => arcsin y - y) (1 / √(1 - y ^ 2) - 1) y :=
    fun y hy => (hasDerivAt_arcsin (by linarith [hy.1]) hy.2.ne).sub (hasDerivAt_id y)
  have hm : MonotoneOn (fun y => arcsin y - y) (Icc 0 1) := by
    refine monotoneOn_of_deriv_nonneg (convex_Icc 0 1)
      (continuous_arcsin.sub continuous_id).continuousOn
      (fun y hy => ?_) (fun y hy => ?_) <;> rw [interior_Icc] at hy
    · exact (hd y hy).differentiableAt.differentiableWithinAt
    · rw [(hd y hy).deriv, sub_nonneg]
      have h0 : 0 < √(1 - y ^ 2) := Real.sqrt_pos.2 (by nlinarith [hy.1, hy.2])
      have h1 : √(1 - y ^ 2) ≤ 1 := Real.sqrt_le_one.2 (by nlinarith)
      rw [le_div_iff₀ h0]; linarith
  have := hm ⟨le_rfl, zero_le_one⟩ ⟨hx0, hx1⟩ hx0
  simp only [arcsin_zero, sub_zero] at this
  linarith

/-- `Ψ(a) ≥ (2/π) a` on `[0, 1]`. -/
theorem Psi_ge {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) : 2 / π * a ≤ Psi a := by
  have := le_arcsin_self ha0 ha1
  have : 0 ≤ a * √(1 - a ^ 2) := by positivity
  have : 0 ≤ 2 / π := by have := Real.pi_pos; positivity
  unfold Psi; nlinarith

/-! ### Conditioning on the first three columns -/

/-- Given `N ≠ 0`, `P(|⟨N, x⟩| < d) = Ψ(min(d/‖N‖, 1))` for `x` uniform on S³ (fact (i)). -/
theorem inner_prob {N : E4} (hN : N ≠ 0) {d : ℝ} (hd : 0 < d) :
    unifS3 {x : S3 | |⟪N, (x : E4)⟫| < d} = ENNReal.ofReal (Psi (min (d / ‖N‖) 1)) := by
  have hn : 0 < ‖N‖ := norm_pos_iff.2 hN
  have hu : ‖‖N‖⁻¹ • N‖ = 1 := by rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hn.ne']
  rw [← unif_abs_inner_lt' hu (div_pos hd hn)]
  congr 1; ext x
  simp only [mem_ofPred_eq, real_inner_smul_left, abs_mul, abs_inv, abs_norm]
  rw [lt_div_iff₀ hn, inv_mul_eq_div, div_mul_cancel₀ _ hn.ne']

/-- **The conditional bound** `P(|⟨N, x⟩| < d) ≤ (4/π) d/‖N‖` (with `1/0 = ∞`). -/
theorem inner_le (N : E4) {d : ℝ} (hd : 0 < d) :
    unifS3 {x : S3 | |⟪N, (x : E4)⟫| < d}
      ≤ ENNReal.ofReal (4 / π * d) * (ENNReal.ofReal ‖N‖)⁻¹ := by
  have hpi := Real.pi_pos
  by_cases hN : N = 0
  · rw [hN, norm_zero, ENNReal.ofReal_zero, ENNReal.inv_zero,
      ENNReal.mul_top (by simp; positivity)]
    exact le_top
  have hn : 0 < ‖N‖ := norm_pos_iff.2 hN
  rw [inner_prob hN hd, ← ENNReal.ofReal_inv_of_pos hn, ← ENNReal.ofReal_mul (by positivity)]
  refine ENNReal.ofReal_le_ofReal ?_
  have hm0 : 0 ≤ min (d / ‖N‖) 1 := le_min (by positivity) zero_le_one
  calc Psi (min (d / ‖N‖) 1) ≤ 4 / π * min (d / ‖N‖) 1 := Psi_le hm0 (min_le_right _ _)
    _ ≤ 4 / π * (d / ‖N‖) := by gcongr; exact min_le_left _ _
    _ = 4 / π * d * ‖N‖⁻¹ := by ring

/-- **The conditional lower bound** `P(|⟨N, x⟩| < d) ≥ Ψ(d)` when `‖N‖ ≤ 1`. -/
theorem inner_ge {N : E4} (hN1 : ‖N‖ ≤ 1) {d : ℝ} (hd : 0 < d) (hd1 : d ≤ 1) :
    ENNReal.ofReal (Psi d) ≤ unifS3 {x : S3 | |⟪N, (x : E4)⟫| < d} := by
  by_cases hN : N = 0
  · have : {x : S3 | |⟪N, (x : E4)⟫| < d} = univ := by
      ext x; simp [hN, hd]
    rw [this, measure_univ, ← ENNReal.ofReal_one, ← Psi_one]
    exact ENNReal.ofReal_le_ofReal (Psi_monoOn ⟨hd.le, hd1⟩ ⟨zero_le_one, le_rfl⟩ hd1)
  have hn : 0 < ‖N‖ := norm_pos_iff.2 hN
  rw [inner_prob hN hd]
  refine ENNReal.ofReal_le_ofReal (Psi_monoOn ⟨hd.le, hd1⟩
    ⟨le_min (by positivity) zero_le_one, min_le_right _ _⟩ (le_min ?_ hd1))
  exact le_div_self hd.le hn hN1

/-! ### `E[1/‖N‖] = 8/π` -/

theorem lint_inv_sqrt :
    ∫⁻ y in Ioo (0 : ℝ) 1, (ENNReal.ofReal √y)⁻¹ * ENNReal.ofReal 1 = 2 := by
  have hr : (-1 : ℝ) < -(1 / 2) := by norm_num
  have hint : IntegrableOn (fun y : ℝ => y ^ (-(1 / 2) : ℝ)) (Ioo 0 1) :=
    ((intervalIntegrable_iff_integrableOn_Ioc_of_le zero_le_one).1
      (intervalIntegral.intervalIntegrable_rpow' hr)).mono_set Ioo_subset_Ioc_self
  rw [setLIntegral_congr_fun measurableSet_Ioo (g := fun y => ENNReal.ofReal (y ^ (-(1 / 2) : ℝ)))
    (fun y hy => by
      rw [ENNReal.ofReal_one, mul_one, ← ENNReal.ofReal_inv_of_pos (Real.sqrt_pos.2 hy.1),
        Real.sqrt_eq_rpow, ← Real.rpow_neg hy.1.le])]
  rw [← ofReal_integral_eq_lintegral_ofReal hint
    (ae_restrict_of_forall_mem measurableSet_Ioo fun y hy => Real.rpow_nonneg hy.1.le _),
    ← integral_Ioc_eq_integral_Ioo, ← intervalIntegral.integral_of_le zero_le_one,
    integral_rpow (Or.inl hr)]
  norm_num

theorem lint_inv_vol2 :
    ∫⁻ y in Ioo (0 : ℝ) 1, (ENNReal.ofReal √(1 - y ^ 2))⁻¹ * ENNReal.ofReal (4 / π * √(1 - y ^ 2))
      = ENNReal.ofReal (4 / π) := by
  have hpi := Real.pi_pos
  rw [setLIntegral_congr_fun measurableSet_Ioo (g := fun _ => ENNReal.ofReal (4 / π))
    (fun y hy => by
      have hs : 0 < √(1 - y ^ 2) := Real.sqrt_pos.2 (by nlinarith [hy.1, hy.2])
      rw [ENNReal.ofReal_mul (by positivity), mul_comm (ENNReal.ofReal (4 / π)), ← mul_assoc,
        ENNReal.inv_mul_cancel (by simp [hs]) ENNReal.ofReal_ne_top, one_mul]),
    setLIntegral_const, Real.volume_Ioo, sub_zero, ENNReal.ofReal_one, mul_one]

theorem measurable_inv_ofReal {α : Type*} [MeasurableSpace α] {f : α → ℝ} (hf : Measurable f) :
    Measurable fun a => (ENNReal.ofReal (f a))⁻¹ :=
  hf.ennreal_ofReal.inv

/-- `E[1/√D] = 2` for the squared distance `D` of fact (ii). -/
theorem lint_distSq {u₁ u₂ : E4} (h₁ : ‖u₁‖ = 1) (h₂ : ‖u₂‖ = 1) (h₁₂ : ⟪u₁, u₂⟫ = 0) :
    ∫⁻ x, (ENNReal.ofReal √(distSq u₁ u₂ x))⁻¹ ∂unifS3 = 2 := by
  have hD : Measurable (distSq u₁ u₂) := by unfold distSq; fun_prop
  have hg : Measurable fun y : ℝ => (ENNReal.ofReal √y)⁻¹ :=
    measurable_inv_ofReal Real.continuous_sqrt.measurable
  rw [← lintegral_map hg hD, law_distSq h₁ h₂ h₁₂, lintegral_densLaw _ measurable_const hg,
    lint_inv_sqrt]

/-- `E[1/√(1 - ⟨n, x⟩²)] = 4/π` (fact (i)). -/
theorem lint_inv_vol2_unit {n : E4} (hn : ‖n‖ = 1) :
    ∫⁻ x, (ENNReal.ofReal √(1 - ⟪n, (x : E4)⟫ ^ 2))⁻¹ ∂unifS3 = ENNReal.ofReal (4 / π) := by
  have hX : Measurable fun x : S3 => |⟪n, (x : E4)⟫| := by fun_prop
  have hg : Measurable fun y : ℝ => (ENNReal.ofReal √(1 - y ^ 2))⁻¹ :=
    measurable_inv_ofReal (by fun_prop)
  have h := lintegral_map (μ := unifS3) hg hX
  simp only [sq_abs] at h
  rw [← h, law_abs_inner hn, lintegral_densLaw _ (by fun_prop) hg, lint_inv_vol2]

/-- Integrating out the third column: `E[1/‖N‖ | x₁, x₂] = 2/√(1 - ⟨x₁, x₂⟩²)`. -/
theorem lint_Nv_x3 {x₁ x₂ : E4} (h₁ : ‖x₁‖ = 1) (h₂ : ‖x₂‖ = 1) :
    ∫⁻ x, (ENNReal.ofReal ‖Nv x₁ x₂ (x : E4)‖)⁻¹ ∂unifS3
      = 2 * (ENNReal.ofReal √(1 - ⟪x₁, x₂⟫ ^ 2))⁻¹ := by
  set a := ⟪x₁, x₂⟫ with ha
  have ha1 : a ^ 2 ≤ 1 := by
    have := abs_real_inner_le_norm x₁ x₂
    rw [h₁, h₂, one_mul] at this
    rw [← sq_abs]; nlinarith [abs_nonneg a]
  rcases eq_or_lt_of_le ha1 with ha1 | ha1
  · -- `x₂ = ±x₁`: `N = 0` for every `x₃`.
    have hN : ∀ x : S3, ‖Nv x₁ x₂ (x : E4)‖ = 0 := fun x => by
      have h := normNv_sq_unit h₁ h₂ (norm_eq_of_mem_sphere x)
      rw [← ha, ha1, sub_self, zero_mul, zero_sub] at h
      nlinarith [sq_nonneg ‖Nv x₁ x₂ (x : E4)‖,
        sq_nonneg (⟪x₂, (x : E4)⟫ - a * ⟪x₁, (x : E4)⟫), norm_nonneg (Nv x₁ x₂ (x : E4))]
    simp only [hN, ENNReal.ofReal_zero, ENNReal.inv_zero, lintegral_const, measure_univ, mul_one,
      ha1, sub_self, Real.sqrt_zero, ENNReal.mul_top two_ne_zero]
  -- `|a| < 1`: an orthonormal pair `x₁, u₂` spanning the plane of `x₁, x₂`.
  set s := √(1 - a ^ 2) with hs
  have hs0 : 0 < s := Real.sqrt_pos.2 (by linarith)
  have hs2 : s ^ 2 = 1 - a ^ 2 := Real.sq_sqrt (by linarith)
  set u₂ := s⁻¹ • (x₂ - a • x₁) with hu
  have e11 : ⟪x₁, x₁⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h₁, one_pow]
  have e22 : ⟪x₂, x₂⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h₂, one_pow]
  have e21 : ⟪x₂, x₁⟫ = a := by rw [real_inner_comm]
  have hu2 : ‖u₂‖ = 1 := by
    have hsq : ‖x₂ - a • x₁‖ ^ 2 = s ^ 2 := by
      rw [← real_inner_self_eq_norm_sq]
      simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right,
        e11, e22, e21, ← ha]
      rw [hs2]; ring
    have hn : ‖x₂ - a • x₁‖ = s := by
      rw [← Real.sqrt_sq (norm_nonneg _), hsq, Real.sqrt_sq hs0.le]
    rw [hu, norm_smul, norm_inv, Real.norm_eq_abs, abs_of_pos hs0, hn, inv_mul_cancel₀ hs0.ne']
  have h12 : ⟪x₁, u₂⟫ = 0 := by
    rw [hu, real_inner_smul_right, inner_sub_right, real_inner_smul_right, e11, ← ha]; ring
  have hfac : ∀ x : S3, ‖Nv x₁ x₂ (x : E4)‖ = s * √(distSq x₁ u₂ x) := fun x => by
    have hN := normNv_sq_unit h₁ h₂ (norm_eq_of_mem_sphere x)
    have hiu : ⟪u₂, (x : E4)⟫ = s⁻¹ * (⟪x₂, (x : E4)⟫ - a * ⟪x₁, (x : E4)⟫) := by
      rw [hu, real_inner_smul_left, inner_sub_left, real_inner_smul_left]
    have hD : s ^ 2 * distSq x₁ u₂ x = ‖Nv x₁ x₂ (x : E4)‖ ^ 2 := by
      rw [hN, distSq, hiu, ← ha, ← hs2]; field_simp
    have hD0 : 0 ≤ distSq x₁ u₂ x := by
      have : 0 ≤ s ^ 2 * distSq x₁ u₂ x := hD ▸ sq_nonneg _
      exact (mul_nonneg_iff_of_pos_left (by positivity : (0 : ℝ) < s ^ 2)).1 this
    rw [← Real.sqrt_sq (norm_nonneg _), ← hD, Real.sqrt_mul (sq_nonneg s), Real.sqrt_sq hs0.le]
  have hmeas : Measurable fun x : S3 => (ENNReal.ofReal √(distSq x₁ u₂ x))⁻¹ :=
    measurable_inv_ofReal (by unfold distSq; fun_prop)
  calc ∫⁻ x, (ENNReal.ofReal ‖Nv x₁ x₂ (x : E4)‖)⁻¹ ∂unifS3
      = ∫⁻ x, (ENNReal.ofReal s)⁻¹ * (ENNReal.ofReal √(distSq x₁ u₂ x))⁻¹ ∂unifS3 := by
        refine lintegral_congr fun x => ?_
        rw [hfac, ENNReal.ofReal_mul hs0.le,
          ENNReal.mul_inv (Or.inl (by simp [hs0])) (Or.inl ENNReal.ofReal_ne_top)]
    _ = (ENNReal.ofReal s)⁻¹ * 2 := by rw [lintegral_const_mul _ hmeas, lint_distSq h₁ hu2 h12]
    _ = 2 * (ENNReal.ofReal √(1 - ⟪x₁, x₂⟫ ^ 2))⁻¹ := by rw [mul_comm]

/-! ### Four independent uniform columns -/

/-- The law of the first three columns. -/
noncomputable def unif3 : Measure ((S3 × S3) × S3) := (unifS3.prod unifS3).prod unifS3

/-- The law of four independent uniform columns `((x₁, x₂), x₃), x₄`. -/
noncomputable def unif4 : Measure (((S3 × S3) × S3) × S3) := unif3.prod unifS3

instance : IsProbabilityMeasure unif3 := by unfold unif3; infer_instance
instance : IsProbabilityMeasure unif4 := by unfold unif4; infer_instance

/-- The cofactor vector of the first three columns. -/
noncomputable def Nq (q : (S3 × S3) × S3) : E4 := Nv q.1.1 q.1.2 q.2

/-- `det M = [x₁ x₂ x₃ x₄]`. -/
noncomputable def detM (p : ((S3 × S3) × S3) × S3) : ℝ :=
  br (WithLp.ofLp (p.1.1.1 : E4)) (WithLp.ofLp (p.1.1.2 : E4)) (WithLp.ofLp (p.1.2 : E4))
    (WithLp.ofLp (p.2 : E4))

theorem detM_eq (p : ((S3 × S3) × S3) × S3) : detM p = ⟪Nq p.1, (p.2 : E4)⟫ :=
  (inner_Nv _ _ _ _).symm

theorem Nv_comp {α : Type*} [TopologicalSpace α] {f g h : α → E4} (hf : Continuous f)
    (hg : Continuous g) (hh : Continuous h) : Continuous fun a => Nv (f a) (g a) (h a) := by
  simp only [Nv, br_expand, ev]
  fun_prop

theorem Nq_continuous : Continuous Nq := by
  unfold Nq
  exact Nv_comp (by fun_prop) (by fun_prop) (by fun_prop)

theorem detM_continuous : Continuous detM := by
  have : detM = fun p => ⟪Nq p.1, (p.2 : E4)⟫ := funext detM_eq
  rw [this]
  exact (Nq_continuous.comp continuous_fst).inner (continuous_subtype_val.comp continuous_snd)

/-- The small-ball event `|det M| < d`. -/
def SB (d : ℝ) : Set (((S3 × S3) × S3) × S3) := {p | |detM p| < d}

theorem SB_measurable (d : ℝ) : MeasurableSet (SB d) :=
  measurableSet_lt (continuous_abs.comp detM_continuous).measurable measurable_const

theorem SB_slice (d : ℝ) (q : (S3 × S3) × S3) :
    Prod.mk q ⁻¹' SB d = {x : S3 | |⟪Nq q, (x : E4)⟫| < d} := by
  ext x; simp only [SB, mem_preimage, mem_ofPred_eq, detM_eq]

theorem measurable_invN : Measurable fun q : (S3 × S3) × S3 => (ENNReal.ofReal ‖Nq q‖)⁻¹ :=
  measurable_inv_ofReal (continuous_norm.comp Nq_continuous).measurable

/-- **`E[1/‖N‖] = 8/π`.** -/
theorem lint_invN : ∫⁻ q, (ENNReal.ofReal ‖Nq q‖)⁻¹ ∂unif3 = ENNReal.ofReal (8 / π) := by
  have hpi := Real.pi_pos
  have hg : Measurable fun p : S3 × S3 =>
      2 * (ENNReal.ofReal √(1 - ⟪(p.1 : E4), (p.2 : E4)⟫ ^ 2))⁻¹ :=
    (measurable_inv_ofReal (by fun_prop)).const_mul 2
  rw [unif3, lintegral_prod _ measurable_invN.aemeasurable]
  simp only [Nq]
  rw [lintegral_congr (g := fun p : S3 × S3 =>
      2 * (ENNReal.ofReal √(1 - ⟪(p.1 : E4), (p.2 : E4)⟫ ^ 2))⁻¹)
    (fun p => lint_Nv_x3 (norm_eq_of_mem_sphere p.1) (norm_eq_of_mem_sphere p.2)),
    lintegral_prod _ hg.aemeasurable]
  simp only
  rw [lintegral_congr (g := fun _ => 2 * ENNReal.ofReal (4 / π)) (fun x₁ => by
      rw [lintegral_const_mul _ (measurable_inv_ofReal (by fun_prop)),
        lint_inv_vol2_unit (norm_eq_of_mem_sphere x₁)]),
    lintegral_const, measure_univ, mul_one, ← ENNReal.ofReal_ofNat 2,
    ← ENNReal.ofReal_mul (by norm_num)]
  congr 1; ring

theorem lint_SB (d : ℝ) :
    unif4 (SB d) = ∫⁻ q, unifS3 {x : S3 | |⟪Nq q, (x : E4)⟫| < d} ∂unif3 := by
  rw [unif4, Measure.prod_apply (SB_measurable d)]
  simp only [SB_slice]

/-- **Proposition 9, upper bound.** `P(|det M| < d) ≤ (32/π²) d`. -/
theorem small_ball_upper {d : ℝ} (hd : 0 < d) : unif4.real (SB d) ≤ 32 / π ^ 2 * d := by
  have hpi := Real.pi_pos
  have h : unif4 (SB d) ≤ ENNReal.ofReal (32 / π ^ 2 * d) := by
    rw [lint_SB]
    calc ∫⁻ q, unifS3 {x : S3 | |⟪Nq q, (x : E4)⟫| < d} ∂unif3
        ≤ ∫⁻ q, ENNReal.ofReal (4 / π * d) * (ENNReal.ofReal ‖Nq q‖)⁻¹ ∂unif3 :=
          lintegral_mono fun q => inner_le _ hd
      _ = ENNReal.ofReal (4 / π * d) * ENNReal.ofReal (8 / π) := by
          rw [lintegral_const_mul _ measurable_invN, lint_invN]
      _ = ENNReal.ofReal (32 / π ^ 2 * d) := by
          rw [← ENNReal.ofReal_mul (by positivity)]; congr 1; field_simp; ring
  exact ENNReal.toReal_le_of_le_ofReal (by positivity) h

/-- **Proposition 9, lower bound.** `(2/π)(d √(1 - d²) + arcsin d) ≤ P(|det M| < d)` for
`0 < d ≤ 1`. -/
theorem small_ball_lower {d : ℝ} (hd : 0 < d) (hd1 : d ≤ 1) :
    2 / π * (d * √(1 - d ^ 2) + arcsin d) ≤ unif4.real (SB d) := by
  have h : ENNReal.ofReal (Psi d) ≤ unif4 (SB d) := by
    rw [lint_SB]
    calc ENNReal.ofReal (Psi d) = ∫⁻ _q, ENNReal.ofReal (Psi d) ∂unif3 := by
          rw [lintegral_const, measure_univ, mul_one]
      _ ≤ _ := lintegral_mono fun q => inner_ge (normNv_le_one (norm_eq_of_mem_sphere _)
          (norm_eq_of_mem_sphere _) (norm_eq_of_mem_sphere _)) hd hd1
  exact (ENNReal.ofReal_le_iff_le_toReal (measure_ne_top _ _)).1 h

/-- The slope of `Ψ` at 0: `Ψ(min(d/r, 1))/d → (4/π)/r` as `d → 0⁺`. -/
theorem Psi_slope {r : ℝ} (hr : 0 < r) :
    Tendsto (fun d => Psi (min (d / r) 1) / d) (𝓝[>] 0) (𝓝 (4 / π / r)) := by
  have hD : HasDerivAt Psi (4 / π) 0 := by
    have := hasDerivAt_Psi (y := 0) (by norm_num) (by norm_num)
    simpa using this
  have hs := hasDerivAt_iff_tendsto_slope_zero.1 hD
  have ht : Tendsto (fun d : ℝ => d / r) (𝓝[>] 0) (𝓝[≠] 0) := by
    refine tendsto_nhdsWithin_iff.2 ⟨?_, ?_⟩
    · have : Tendsto (fun d : ℝ => d / r) (𝓝 0) (𝓝 (0 / r)) :=
        (continuous_id.div_const r).tendsto 0
      rw [zero_div] at this
      exact this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with d hd
      exact div_ne_zero (ne_of_gt hd) hr.ne'
  have hc := (hs.comp ht).mul_const r⁻¹
  rw [← div_eq_mul_inv] at hc
  refine hc.congr' ?_
  filter_upwards [Ioo_mem_nhdsGT hr] with d hd
  have hdr : d / r < 1 := (div_lt_one hr).2 hd.2
  simp only [Function.comp_apply, zero_add, Psi_zero, sub_zero, smul_eq_mul,
    min_eq_left hdr.le]
  field_simp

/-- **Proposition 9, the limit.** `P(|det M| < d)/d → 32/π²` as `d → 0⁺`. -/
theorem small_ball_limit :
    Tendsto (fun d => unif4.real (SB d) / d) (𝓝[>] 0) (𝓝 (32 / π ^ 2)) := by
  have hpi := Real.pi_pos
  set bound : (S3 × S3) × S3 → ℝ≥0∞ := fun q =>
    ENNReal.ofReal (4 / π) * (ENNReal.ofReal ‖Nq q‖)⁻¹
  have hbint : ∫⁻ q, bound q ∂unif3 = ENNReal.ofReal (32 / π ^ 2) := by
    rw [lintegral_const_mul _ measurable_invN, lint_invN, ← ENNReal.ofReal_mul (by positivity)]
    congr 1; field_simp; ring
  have hpos : ∀ᵐ q ∂unif3, 0 < ‖Nq q‖ := by
    filter_upwards [ae_lt_top measurable_invN (by rw [lint_invN]; exact ENNReal.ofReal_ne_top)]
      with q hq
    rw [ENNReal.inv_lt_top, ENNReal.ofReal_pos] at hq
    exact hq
  set F : ℝ → (S3 × S3) × S3 → ℝ≥0∞ := fun d q =>
    unifS3 {x : S3 | |⟪Nq q, (x : E4)⟫| < d} * (ENNReal.ofReal d)⁻¹
  have hF : ∀ d, 0 < d → unif4 (SB d) * (ENNReal.ofReal d)⁻¹ = ∫⁻ q, F d q ∂unif3 := by
    intro d hd
    rw [lint_SB, lintegral_mul_const]
    have := measurable_measure_prodMk_left (ν := unifS3) (SB_measurable d)
    simpa only [SB_slice] using this
  have hlim : Tendsto (fun d => ∫⁻ q, F d q ∂unif3) (𝓝[>] 0) (𝓝 (ENNReal.ofReal (32 / π ^ 2))) := by
    rw [← hbint]
    refine tendsto_lintegral_filter_of_dominated_convergence bound ?_ ?_ ?_ ?_
    · refine Eventually.of_forall fun d => ?_
      have := measurable_measure_prodMk_left (ν := unifS3) (SB_measurable d)
      simp only [SB_slice] at this
      exact this.mul_const _
    · filter_upwards [self_mem_nhdsWithin] with d (hd : 0 < d)
      refine Eventually.of_forall fun q => ?_
      calc F d q ≤ ENNReal.ofReal (4 / π * d) * (ENNReal.ofReal ‖Nq q‖)⁻¹ * (ENNReal.ofReal d)⁻¹ :=
            mul_le_mul_left (inner_le _ hd) _
        _ = bound q * (ENNReal.ofReal d * (ENNReal.ofReal d)⁻¹) := by
            rw [ENNReal.ofReal_mul (by positivity)]; ring
        _ = bound q := by
            rw [ENNReal.mul_inv_cancel (by simp [hd]) ENNReal.ofReal_ne_top, mul_one]
    · rw [hbint]; exact ENNReal.ofReal_ne_top
    · filter_upwards [hpos] with q hq
      have hN : Nq q ≠ 0 := norm_pos_iff.1 hq
      have h := ENNReal.tendsto_ofReal (Psi_slope hq)
      rw [ENNReal.ofReal_div_of_pos hq, div_eq_mul_inv] at h
      refine h.congr' ?_
      filter_upwards [self_mem_nhdsWithin] with d (hd : 0 < d)
      rw [ENNReal.ofReal_div_of_pos hd, div_eq_mul_inv, ← inner_prob hN hd]
  have hlim' := (ENNReal.tendsto_toReal ENNReal.ofReal_ne_top).comp hlim
  rw [ENNReal.toReal_ofReal (by positivity)] at hlim'
  refine hlim'.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with d (hd : 0 < d)
  rw [Function.comp_apply, ← hF d hd, ENNReal.toReal_mul, ENNReal.toReal_inv,
    ENNReal.toReal_ofReal hd.le, measureReal_def, div_eq_mul_inv]

/-! ### Six independent uniform points: tails of κ_J and L(S) -/

section Six

open ProbabilityTheory

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
  {X : Fin 6 → Ω → S3}

/-- The configuration of the six points. -/
noncomputable def cfg (X : Fin 6 → Ω → S3) (ω : Ω) : Config 6 := fun m => WithLp.ofLp (X m ω : E4)

omit [MeasurableSpace Ω] in
theorem cfg_unit (ω : Ω) (m : Fin 6) : ‖WithLp.toLp 2 (cfg X ω m)‖ = 1 := by
  simp only [cfg, WithLp.toLp_ofLp]; exact norm_eq_of_mem_sphere _

/-- **Marginal law.** Four distinct points of six i.i.d. uniform points have the law `unif4`. -/
theorem map_four (hX : ∀ i, Measurable (X i)) (hind : iIndepFun X P)
    (hlaw : ∀ i, P.map (X i) = unifS3) {a b c e : Fin 6} (hab : a ≠ b) (hac : a ≠ c)
    (hae : a ≠ e) (hbc : b ≠ c) (hbe : b ≠ e) (hce : c ≠ e) :
    P.map (fun ω => (((X a ω, X b ω), X c ω), X e ω)) = unif4 := by
  have h2 : P.map (fun ω => (X a ω, X b ω)) = unifS3.prod unifS3 := by
    rw [(hind.indepFun hab).map_prod_eq_prod_map_map (hX a).aemeasurable (hX b).aemeasurable,
      hlaw, hlaw]
  have m2 : Measurable fun ω => (X a ω, X b ω) := (hX a).prodMk (hX b)
  have i3 : IndepFun (fun ω => (X a ω, X b ω)) (X c) P := by
    have h := hind.indepFun_finset {a, b} {c}
      (Finset.disjoint_singleton_right.2 (by simp [hac.symm, hbc.symm])) hX
    have ha : a ∈ ({a, b} : Finset (Fin 6)) := by simp
    have hb : b ∈ ({a, b} : Finset (Fin 6)) := by simp
    have hc : c ∈ ({c} : Finset (Fin 6)) := by simp
    have hψ : Measurable fun g : ({c} : Finset (Fin 6)) → S3 => g ⟨c, hc⟩ := measurable_pi_apply _
    have hφ : Measurable fun g : ({a, b} : Finset (Fin 6)) → S3 => (g ⟨a, ha⟩, g ⟨b, hb⟩) :=
      (measurable_pi_apply _).prodMk (measurable_pi_apply _)
    exact h.comp hφ hψ
  have h3 : P.map (fun ω => ((X a ω, X b ω), X c ω)) = (unifS3.prod unifS3).prod unifS3 := by
    rw [i3.map_prod_eq_prod_map_map m2.aemeasurable (hX c).aemeasurable, h2, hlaw]
  have m3 : Measurable fun ω => ((X a ω, X b ω), X c ω) := m2.prodMk (hX c)
  have i4 : IndepFun (fun ω => ((X a ω, X b ω), X c ω)) (X e) P := by
    have h := hind.indepFun_finset {a, b, c} {e}
      (Finset.disjoint_singleton_right.2 (by simp [hae.symm, hbe.symm, hce.symm])) hX
    have ha : a ∈ ({a, b, c} : Finset (Fin 6)) := by simp
    have hb : b ∈ ({a, b, c} : Finset (Fin 6)) := by simp
    have hc : c ∈ ({a, b, c} : Finset (Fin 6)) := by simp
    have he : e ∈ ({e} : Finset (Fin 6)) := by simp
    have hψ : Measurable fun g : ({e} : Finset (Fin 6)) → S3 => g ⟨e, he⟩ := measurable_pi_apply _
    have hφ : Measurable fun g : ({a, b, c} : Finset (Fin 6)) → S3 =>
        ((g ⟨a, ha⟩, g ⟨b, hb⟩), g ⟨c, hc⟩) :=
      ((measurable_pi_apply _).prodMk (measurable_pi_apply _)).prodMk (measurable_pi_apply _)
    exact h.comp hφ hψ
  rw [i4.map_prod_eq_prod_map_map m3.aemeasurable (hX e).aemeasurable, h3, hlaw]
  rfl

/-- Every bracket of four distinct points of the six has the small-ball bound of
Proposition 9. -/
theorem bracket_small_ball (hX : ∀ i, Measurable (X i)) (hind : iIndepFun X P)
    (hlaw : ∀ i, P.map (X i) = unifS3) {q : Fin 4 → Fin 6} (hq : Function.Injective q)
    {d : ℝ} (hd : 0 < d) :
    P.real {ω | |brf (q 0) (q 1) (q 2) (q 3) (cfg X ω)| < d} ≤ 32 / π ^ 2 * d := by
  have hne : ∀ i j : Fin 4, i ≠ j → q i ≠ q j := fun i j h e => h (hq e)
  have hmap := map_four hX hind hlaw (hne 0 1 (by decide)) (hne 0 2 (by decide))
    (hne 0 3 (by decide)) (hne 1 2 (by decide)) (hne 1 3 (by decide)) (hne 2 3 (by decide))
  have hg : Measurable fun ω => (((X (q 0) ω, X (q 1) ω), X (q 2) ω), X (q 3) ω) :=
    (((hX _).prodMk (hX _)).prodMk (hX _)).prodMk (hX _)
  have hset : {ω | |brf (q 0) (q 1) (q 2) (q 3) (cfg X ω)| < d}
      = (fun ω => (((X (q 0) ω, X (q 1) ω), X (q 2) ω), X (q 3) ω)) ⁻¹' SB d := rfl
  rw [hset, measureReal_def, ← Measure.map_apply hg (SB_measurable d), hmap, ← measureReal_def]
  exact small_ball_upper hd

/-- The 11 distinct brackets used by J1-J3. -/
def Q11 : Fin 11 → Fin 4 → Fin 6 :=
  ![![0, 1, 2, 3], ![0, 1, 4, 5], ![0, 1, 2, 4], ![0, 1, 3, 5], ![0, 2, 3, 4], ![1, 2, 3, 5],
    ![0, 2, 3, 5], ![1, 2, 3, 4], ![2, 3, 4, 5], ![0, 2, 4, 5], ![1, 3, 4, 5]]

theorem Q11_injective (m : Fin 11) : Function.Injective (Q11 m) := by
  revert m; decide

theorem Jq_mem_Q11 (k : Fin 3) (i : Fin 4) : ∃ m, Q11 m = Jq k i := by
  revert k i; decide

/-- **Tail of κ_J for six i.i.d. uniform points of S³.** `P(κ_J > M) ≤ 2816/(π² M)`. -/
theorem kappa_tail_six (hX : ∀ i, Measurable (X i)) (hind : iIndepFun X P)
    (hlaw : ∀ i, P.map (X i) = unifS3) {M : ℝ} (hM : 0 < M) :
    P.real {ω | M < kappaJ (cfg X ω)} ≤ 2816 / (π ^ 2 * M) := by
  have h := kappa_tail P (fun m ω => brf (Q11 m 0) (Q11 m 1) (Q11 m 2) (Q11 m 3) (cfg X ω))
    (fun ω => kappaJ (cfg X ω)) (C := 32 / π ^ 2)
    (fun m d hd => bracket_small_ball hX hind hlaw (Q11_injective m) hd) ?_ hM
  · calc _ ≤ 11 * (32 / π ^ 2) * (8 / M) := h
      _ = 2816 / (π ^ 2 * M) := by field_simp; ring
  intro ω
  obtain ⟨ki, -, hki⟩ := Finset.exists_mem_eq_inf' Finset.univ_nonempty
    (fun ki : Fin 3 × Fin 4 => |bJ ki.1 ki.2 (cfg X ω)|)
  have hδ : deltaJ (cfg X ω) = |bJ ki.1 ki.2 (cfg X ω)| := hki
  obtain ⟨m, hm⟩ := Jq_mem_Q11 ki.1 ki.2
  refine ⟨m, ?_⟩
  have hb : brf (Q11 m 0) (Q11 m 1) (Q11 m 2) (Q11 m 3) (cfg X ω) = bJ ki.1 ki.2 (cfg X ω) := by
    rw [hm]; rfl
  simp only [hb, ← hδ]
  rcases lt_or_ge 0 (deltaJ (cfg X ω)) with hpos | hle
  · have := kappa_le (fun m => cfg_unit ω m) hpos
    rwa [le_div_iff₀ hpos] at this
  · have : deltaJ (cfg X ω) = 0 := le_antisymm hle (hδ ▸ abs_nonneg _)
    rw [this, mul_zero]; norm_num

/-! #### L(S) = 8/δ_min² -/

/-- `L(S) = 8/δ_min²`. -/
noncomputable def Lsix (C : Config 6) : ℝ := 8 / deltaMin C ^ 2

/-- The 15 four-point subsets of six points, in increasing order. -/
def Q15 : Fin 15 → Fin 4 → Fin 6 :=
  ![![0, 1, 2, 3], ![0, 1, 2, 4], ![0, 1, 2, 5], ![0, 1, 3, 4], ![0, 1, 3, 5], ![0, 1, 4, 5],
    ![0, 2, 3, 4], ![0, 2, 3, 5], ![0, 2, 4, 5], ![0, 3, 4, 5], ![1, 2, 3, 4], ![1, 2, 3, 5],
    ![1, 2, 4, 5], ![1, 3, 4, 5], ![2, 3, 4, 5]]

theorem Q15_injective (m : Fin 15) : Function.Injective (Q15 m) := by
  revert m; decide

theorem Q15_cover : ∀ s : Finset (Fin 6), s.card = 4 → ∃ m, ∀ i, Q15 m i ∈ s := by
  decide

/-- A bracket changes at most its sign when its four points are reordered. -/
theorem abs_br_perm (v : Fin 4 → Fin 4 → ℝ) (σ : Equiv.Perm (Fin 4)) :
    |br (v (σ 0)) (v (σ 1)) (v (σ 2)) (v (σ 3))| = |br (v 0) (v 1) (v 2) (v 3)| := by
  have h1 : Matrix.of ![v (σ 0), v (σ 1), v (σ 2), v (σ 3)] = (Matrix.of v).submatrix σ id := by
    ext i j; fin_cases i <;> rfl
  have h2 : Matrix.of ![v 0, v 1, v 2, v 3] = Matrix.of v := by
    ext i j; fin_cases i <;> rfl
  simp only [br, h1, h2, Matrix.det_permute]
  rcases Int.units_eq_one_or (Equiv.Perm.sign σ) with h | h <;> simp [h, abs_neg]

/-- Every quadruple of distinct points is a reordering of one of the 15 subsets. -/
theorem injective_eq_Q15 {q : Fin 4 → Fin 6} (hq : Function.Injective q) :
    ∃ m, ∃ σ : Equiv.Perm (Fin 4), ∀ i, q i = Q15 m (σ i) := by
  classical
  set s := Finset.univ.image q
  have hs : s.card = 4 := by rw [Finset.card_image_of_injective _ hq]; simp
  obtain ⟨m, hm⟩ := Q15_cover s hs
  have himg : Finset.univ.image (Q15 m) = s := by
    refine Finset.eq_of_subset_of_card_le (fun x hx => ?_) ?_
    · obtain ⟨i, -, rfl⟩ := Finset.mem_image.1 hx; exact hm i
    · rw [hs, Finset.card_image_of_injective _ (Q15_injective m)]; simp
  have hpre : ∀ i, ∃ j, Q15 m j = q i := fun i => by
    have : q i ∈ Finset.univ.image (Q15 m) := himg ▸ Finset.mem_image_of_mem q (Finset.mem_univ i)
    obtain ⟨j, -, hj⟩ := Finset.mem_image.1 this
    exact ⟨j, hj⟩
  choose τ hτ using hpre
  have hτi : Function.Injective τ := fun i j h => hq (by rw [← hτ i, ← hτ j, h])
  refine ⟨m, Equiv.ofBijective τ (Finite.injective_iff_bijective.1 hτi), fun i => ?_⟩
  simp [hτ]

theorem deltaMin_eq (C : Config 6) :
    ∃ m, deltaMin C = |brf (Q15 m 0) (Q15 m 1) (Q15 m 2) (Q15 m 3) C| := by
  obtain ⟨q, hq, hqe⟩ := Finset.exists_mem_eq_inf'
    (⟨![0, 1, 2, 3], by simp; decide⟩ :
      (Finset.univ.filter fun q : Fin 4 → Fin 6 => Function.Injective q).Nonempty)
    (fun q => |brf (q 0) (q 1) (q 2) (q 3) C|)
  obtain ⟨m, σ, hσ⟩ := injective_eq_Q15 (Finset.mem_filter.1 hq).2
  refine ⟨m, ?_⟩
  rw [deltaMin, hqe]
  simp only [brf, hσ]
  exact abs_br_perm (fun i => C (Q15 m i)) σ

theorem deltaMin_le (C : Config 6) :
    deltaMin C ≤ |brf (Q15 0 0) (Q15 0 1) (Q15 0 2) (Q15 0 3) C| :=
  Finset.inf'_le (fun q : Fin 4 → Fin 6 => |brf (q 0) (q 1) (q 2) (q 3) C|)
    (Finset.mem_filter.2 ⟨Finset.mem_univ _, Q15_injective 0⟩)

/-- **Tail of L(S) for six i.i.d. uniform points of S³.**
`P(L(S) > l) ≤ 15 (32/π²) √(8/l)` for `l > 0`, and `P(L(S) > l) ≥ (2/π) √(8/l)` for `l ≥ 8`:
`P(L(S) > l) = Θ(l^{-1/2})`. -/
theorem L_tail_six (hX : ∀ i, Measurable (X i)) (hind : iIndepFun X P)
    (hlaw : ∀ i, P.map (X i) = unifS3) {l : ℝ} (hl : 0 < l) :
    P.real {ω | l < Lsix (cfg X ω)} ≤ 15 * (32 / π ^ 2) * √(8 / l) ∧
    (8 ≤ l → 2 / π * √(8 / l) ≤ P.real {ω | l < Lsix (cfg X ω)}) := by
  have hpi := Real.pi_pos
  set b : Fin 15 → Ω → ℝ := fun m ω => brf (Q15 m 0) (Q15 m 1) (Q15 m 2) (Q15 m 3) (cfg X ω)
  have hsb : ∀ m d, 0 < d → P.real {ω | |b m ω| < d} ≤ 32 / π ^ 2 * d := fun m d hd =>
    bracket_small_ball hX hind hlaw (Q15_injective m) hd
  have hx : 0 < √(8 / l) := Real.sqrt_pos.2 (by positivity)
  constructor
  · have hsub : {ω | l < Lsix (cfg X ω)} ⊆ ⋃ m, {ω | |b m ω| < √(8 / l)} := by
      intro ω hω
      simp only [mem_ofPred_eq, Lsix] at hω
      obtain ⟨m, hm⟩ := deltaMin_eq (cfg X ω)
      simp only [mem_iUnion, mem_ofPred_eq]
      refine ⟨m, ?_⟩
      have hδ0 : 0 ≤ deltaMin (cfg X ω) := hm ▸ abs_nonneg _
      rcases eq_or_lt_of_le hδ0 with h0 | hpos
      · rw [← h0] at hω; norm_num at hω; linarith
      · rw [← hm, Real.lt_sqrt hδ0]
        rw [lt_div_iff₀ (by positivity)] at hω
        rw [lt_div_iff₀ hl]; linarith
    calc P.real {ω | l < Lsix (cfg X ω)} ≤ P.real (⋃ m, {ω | |b m ω| < √(8 / l)}) :=
          measureReal_mono hsub
      _ ≤ ∑ m, P.real {ω | |b m ω| < √(8 / l)} := measureReal_iUnion_fintype_le _
      _ ≤ ∑ _m : Fin 15, 32 / π ^ 2 * √(8 / l) := Finset.sum_le_sum fun m _ => hsb m _ hx
      _ = 15 * (32 / π ^ 2) * √(8 / l) := by simp; ring
  · intro hl8
    have hx1 : √(8 / l) ≤ 1 := Real.sqrt_le_one.2 ((div_le_one hl).2 hl8)
    -- the null event `δ_min = 0`
    have hnull : ∀ m, P {ω | b m ω = 0} = 0 := by
      intro m
      have hle : ∀ ε, 0 < ε → P.real {ω | b m ω = 0} ≤ 32 / π ^ 2 * ε := fun ε hε =>
        le_trans (measureReal_mono fun ω (h : b m ω = 0) => by
          simp only [mem_ofPred_eq, h, abs_zero]; exact hε) (hsb m ε hε)
      have h0 : P.real {ω | b m ω = 0} ≤ 0 := by
        by_contra hc; push Not at hc
        have := hle (P.real {ω | b m ω = 0} * π ^ 2 / 64) (by positivity)
        field_simp at this; nlinarith
      have h0' := le_antisymm h0 measureReal_nonneg
      rwa [measureReal_eq_zero_iff] at h0'
    have hZ : P (⋃ m, {ω | b m ω = 0}) = 0 := measure_iUnion_null hnull
    have hsub : {ω | |b 0 ω| < √(8 / l)} ⊆ {ω | l < Lsix (cfg X ω)} ∪ ⋃ m, {ω | b m ω = 0} := by
      intro ω hω
      simp only [mem_ofPred_eq] at hω
      obtain ⟨m, hm⟩ := deltaMin_eq (cfg X ω)
      rcases eq_or_lt_of_le (hm ▸ abs_nonneg _ : 0 ≤ deltaMin (cfg X ω)) with h0 | hpos
      · right; simp only [mem_iUnion, mem_ofPred_eq]
        exact ⟨m, abs_eq_zero.1 (hm ▸ h0.symm)⟩
      · left
        simp only [mem_ofPred_eq, Lsix]
        have hle := deltaMin_le (cfg X ω)
        have h1 : deltaMin (cfg X ω) < √(8 / l) := lt_of_le_of_lt hle hω
        have h2 : deltaMin (cfg X ω) ^ 2 < 8 / l := by
          have := pow_lt_pow_left₀ h1 hpos.le two_ne_zero
          rwa [Real.sq_sqrt (by positivity)] at this
        rw [lt_div_iff₀ (by positivity)]
        rw [lt_div_iff₀ hl] at h2; linarith
    have hlow : Psi (√(8 / l)) ≤ P.real {ω | |b 0 ω| < √(8 / l)} := by
      have hmap := map_four hX hind hlaw (a := Q15 0 0) (b := Q15 0 1) (c := Q15 0 2)
        (e := Q15 0 3) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
      have hg : Measurable fun ω =>
          (((X (Q15 0 0) ω, X (Q15 0 1) ω), X (Q15 0 2) ω), X (Q15 0 3) ω) :=
        (((hX _).prodMk (hX _)).prodMk (hX _)).prodMk (hX _)
      have hset : {ω | |b 0 ω| < √(8 / l)} = (fun ω =>
          (((X (Q15 0 0) ω, X (Q15 0 1) ω), X (Q15 0 2) ω), X (Q15 0 3) ω)) ⁻¹' SB (√(8 / l)) :=
        rfl
      rw [hset, measureReal_def, ← Measure.map_apply hg (SB_measurable _), hmap,
        ← measureReal_def]
      exact small_ball_lower hx hx1
    calc 2 / π * √(8 / l) ≤ Psi (√(8 / l)) := Psi_ge hx.le hx1
      _ ≤ P.real {ω | |b 0 ω| < √(8 / l)} := hlow
      _ ≤ P.real ({ω | l < Lsix (cfg X ω)} ∪ ⋃ m, {ω | b m ω = 0}) := measureReal_mono hsub
      _ ≤ P.real {ω | l < Lsix (cfg X ω)} + P.real (⋃ m, {ω | b m ω = 0}) :=
          measureReal_union_le _ _
      _ = P.real {ω | l < Lsix (cfg X ω)} := by
          rw [show P.real (⋃ m, {ω | b m ω = 0}) = 0 by simp [measureReal_def, hZ], add_zero]

end Six

end Capri
