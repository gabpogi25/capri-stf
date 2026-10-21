/-
Parts of Proposition 9 (MATHEMATICS.md Proposition 5.4, small ball for four uniform points of S³).

* `integral_sqrt_one_sub_sq_sym`: `∫_{-d}^{d} √(1 - t²) dt = d √(1 - d²) + arcsin d`, so the
  lower bound `(2/π)(d √(1 - d²) + arcsin d)` is the integral of the density `(2/π) √(1 - t²)`
  of fact (i) over `(-d, d)`; at `d = 1` the density integrates to 1.
* `withDensity_Ioo_le`: a law with density at most `c` gives `(-a, a)` mass at most `2ac`
  (the step `P(|t| < d/|N| | x₁, x₂, x₃) ≤ (4/π) d/|N|`).
* `mean_inv_vol2`, `mean_inv_dist`, `small_ball_constant`: `E[1/vol₂] = 4/π` and
  `E[1/dist] = 2` from the densities of facts (i) and (ii), and the constant
  `(4/π) E[1/vol₃] = 32/π²`.
* `kappa_tail`: `P(κ_J > M) ≤ 11 · C · 8/M` from a small-ball bound `P(|b| < d) ≤ C d` for each
  of the 11 brackets used by J1-J3 and `κ_J ≤ 8/δ_J`; with `C = 32/π²` this is `2816/(π² M)`.
* `L_tail`: `P(|b₁| < √(8/l)) ≤ P(L(S) > l) ≤ 15 · C · √(8/l)`.

The laws of facts (i) and (ii), the conditioning that gives `E[1/vol₃] = 8/π`, the bounds, the
limit `P(|det M| < d)/d → 32/π²` and both tails for six independent uniform points are proved in
`Sphere.lean`, `SphereLaw.lean` and `SphereSmallBall.lean`.
-/
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.MeasureTheory.Measure.WithDensity

open MeasureTheory Real intervalIntegral Set
open scoped ENNReal Interval

namespace Capri

/-- `∫_{-d}^{d} √(1 - t²) dt = d √(1 - d²) + arcsin d` for `0 ≤ d ≤ 1`. -/
theorem integral_sqrt_one_sub_sq_sym {d : ℝ} (hd0 : 0 ≤ d) (hd1 : d ≤ 1) :
    ∫ x in (-d)..d, √(1 - x ^ 2) = d * √(1 - d ^ 2) + arcsin d := by
  set θ := arcsin d with hθ
  have hs : sin θ = d := sin_arcsin (by linarith) hd1
  have hθ0 : 0 ≤ θ := arcsin_nonneg.2 hd0
  have hθ1 : θ ≤ π / 2 := arcsin_le_pi_div_two d
  calc ∫ x in (-d)..d, √(1 - x ^ 2)
      = ∫ x in sin (-θ)..sin θ, √(1 - x ^ 2) := by rw [sin_neg, hs]
    _ = ∫ x in (-θ)..θ, √(1 - sin x ^ 2) * cos x :=
        (integral_comp_mul_deriv (fun x _ => hasDerivAt_sin x) continuousOn_cos
          (by fun_prop)).symm
    _ = ∫ x in (-θ)..θ, cos x ^ 2 := by
        refine integral_congr_ae (MeasureTheory.ae_of_all _ fun x h => ?_)
        rw [uIoc_of_le (by linarith), Set.mem_Ioc] at h
        rw [← Real.cos_eq_sqrt_one_sub_sin_sq (by linarith [h.1]) (by linarith [h.2]), pow_two]
    _ = d * √(1 - d ^ 2) + θ := by
        rw [integral_cos_sq, cos_neg, sin_neg, hs, hθ, cos_arcsin]
        ring

/-- The density `(2/π) √(1 - t²)` of fact (i) gives `(-d, d)` the mass
`(2/π)(d √(1 - d²) + arcsin d)`, the lower bound of Proposition 9; at `d = 1` it is 1. -/
theorem small_ball_lower_integral {d : ℝ} (hd0 : 0 ≤ d) (hd1 : d ≤ 1) :
    ∫ x in (-d)..d, 2 / π * √(1 - x ^ 2) = 2 / π * (d * √(1 - d ^ 2) + arcsin d) ∧
    ∫ x in (-1 : ℝ)..1, 2 / π * √(1 - x ^ 2) = 1 := by
  refine ⟨by rw [intervalIntegral.integral_const_mul, integral_sqrt_one_sub_sq_sym hd0 hd1], ?_⟩
  rw [intervalIntegral.integral_const_mul, integral_sqrt_one_sub_sq]
  field_simp

/-- A law on ℝ with a density bounded by `c` gives the interval `(-a, a)` mass at most `2ac`. -/
theorem withDensity_Ioo_le (f : ℝ → ℝ≥0∞) {c : ℝ≥0∞} (hf : ∀ x, f x ≤ c) (a : ℝ) :
    volume.withDensity f (Ioo (-a) a) ≤ c * ENNReal.ofReal (2 * a) := by
  rw [withDensity_apply _ measurableSet_Ioo]
  calc ∫⁻ x in Ioo (-a) a, f x ≤ ∫⁻ _ in Ioo (-a) a, c :=
        setLIntegral_mono measurable_const fun x _ => hf x
    _ = c * ENNReal.ofReal (2 * a) := by
        rw [setLIntegral_const, Real.volume_Ioo]; congr 2; ring

/-- `E[1/vol₂(x₁, x₂)] = 4/π`: `vol₂ = √(1 - t²)` with `t = ⟨x₁, x₂⟩` of density
`(2/π) √(1 - t²)` (fact (i)). -/
theorem mean_inv_vol2 : ∫ t in (-1 : ℝ)..1, 2 / π * √(1 - t ^ 2) * (√(1 - t ^ 2))⁻¹ = 4 / π := by
  have h : ∀ t ∈ Ι (-1 : ℝ) 1, t ≠ 1 → 2 / π * √(1 - t ^ 2) * (√(1 - t ^ 2))⁻¹ = 2 / π := by
    intro t ht h1
    rw [uIoc_of_le (by norm_num), Set.mem_Ioc] at ht
    have hlt : t < 1 := lt_of_le_of_ne ht.2 h1
    have hpos : 0 < √(1 - t ^ 2) := Real.sqrt_pos.2 (by nlinarith)
    field_simp
  have hae : ∀ᵐ t ∂volume, t ∈ Ι (-1 : ℝ) 1 →
      2 / π * √(1 - t ^ 2) * (√(1 - t ^ 2))⁻¹ = (fun _ => 2 / π) t := by
    have h1 : ∀ᵐ t ∂(volume : Measure ℝ), t ≠ 1 := by
      simp [ae_iff, measure_singleton]
    filter_upwards [h1] with t ht1 ht using h t ht ht1
  rw [intervalIntegral.integral_congr_ae hae, intervalIntegral.integral_const]
  simp only [smul_eq_mul]; ring

/-- `E[1/dist] = 2`: the distance of `x₃` to the plane of `x₁, x₂` has density `2ρ` on
`[0, 1]` (fact (ii)). -/
theorem mean_inv_dist : ∫ ρ in (0 : ℝ)..1, 2 * ρ * ρ⁻¹ = 2 := by
  have hae : ∀ᵐ ρ ∂volume, ρ ∈ Ι (0 : ℝ) 1 → 2 * ρ * ρ⁻¹ = (fun _ => (2 : ℝ)) ρ := by
    refine MeasureTheory.ae_of_all _ fun ρ hρ => ?_
    rw [uIoc_of_le (by norm_num), Set.mem_Ioc] at hρ
    have : ρ ≠ 0 := ne_of_gt hρ.1
    field_simp
  rw [intervalIntegral.integral_congr_ae hae, intervalIntegral.integral_const]
  simp

/-- The constant of Proposition 9: `(4/π) · E[1/vol₃]` with `E[1/vol₃] = E[1/vol₂] E[1/dist]`
is `32/π²`, and `11 · (32/π²) · 8 = 2816/π²`. -/
theorem small_ball_constant :
    4 / π * ((∫ t in (-1 : ℝ)..1, 2 / π * √(1 - t ^ 2) * (√(1 - t ^ 2))⁻¹) *
      ∫ ρ in (0 : ℝ)..1, 2 * ρ * ρ⁻¹) = 32 / π ^ 2 ∧
    11 * (32 / π ^ 2) * 8 = 2816 / π ^ 2 := by
  rw [mean_inv_vol2, mean_inv_dist]
  constructor <;> ring

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Tail of κ_J.** If `κ ≤ 8/δ_J`, with `δ_J` the smallest of the 11 brackets used by J1-J3
(stated as: some used bracket has `κ |b_k| ≤ 8`), and each bracket has the small-ball bound
`P(|b_k| < d) ≤ C d`, then `P(κ > M) ≤ 11 · C · 8/M`. -/
theorem kappa_tail (P : Measure Ω) [IsFiniteMeasure P] (b : Fin 11 → Ω → ℝ) (κ : Ω → ℝ)
    {C : ℝ} (hsb : ∀ k d, 0 < d → P.real {ω | |b k ω| < d} ≤ C * d)
    (hκ : ∀ ω, ∃ k, κ ω * |b k ω| ≤ 8) {M : ℝ} (hM : 0 < M) :
    P.real {ω | M < κ ω} ≤ 11 * C * (8 / M) := by
  have hsub : {ω | M < κ ω} ⊆ ⋃ k, {ω | |b k ω| < 8 / M} := by
    intro ω hω
    obtain ⟨k, hk⟩ := hκ ω
    simp only [Set.mem_ofPred_eq, Set.mem_iUnion] at hω ⊢
    refine ⟨k, ?_⟩
    rw [lt_div_iff₀ hM]
    have hb := abs_nonneg (b k ω)
    by_contra h
    push Not at h
    nlinarith
  calc P.real {ω | M < κ ω} ≤ P.real (⋃ k, {ω | |b k ω| < 8 / M}) := measureReal_mono hsub
    _ ≤ ∑ k, P.real {ω | |b k ω| < 8 / M} := measureReal_iUnion_fintype_le _
    _ ≤ ∑ _k : Fin 11, C * (8 / M) := Finset.sum_le_sum fun k _ => hsb k _ (by positivity)
    _ = 11 * C * (8 / M) := by simp; ring

/-- **Tail of L(S).** If `L(S) = 8/δ_min²` (stated as: `8 ≤ L |b_k|²` for all 15 brackets, with
equality for one) and each bracket has `P(|b_k| < d) ≤ C d`, then
`P(|b₀| < √(8/l)) ≤ P(L > l) ≤ 15 · C · √(8/l)`. With the lower bound of
`small_ball_lower_integral` this gives `P(L(S) > l) = Θ(l^{-1/2})`. -/
theorem L_tail (P : Measure Ω) [IsFiniteMeasure P] (b : Fin 15 → Ω → ℝ) (L : Ω → ℝ)
    {C : ℝ} (hsb : ∀ k d, 0 < d → P.real {ω | |b k ω| < d} ≤ C * d)
    (hL : ∀ ω, 0 < L ω ∧ (∀ k, 8 ≤ L ω * |b k ω| ^ 2) ∧ ∃ k, L ω * |b k ω| ^ 2 = 8)
    {l : ℝ} (hl : 0 < l) :
    P.real {ω | |b 0 ω| < √(8 / l)} ≤ P.real {ω | l < L ω} ∧
    P.real {ω | l < L ω} ≤ 15 * C * √(8 / l) := by
  have hpos : 0 < √(8 / l) := Real.sqrt_pos.2 (by positivity)
  constructor
  · refine measureReal_mono fun ω hω => ?_
    simp only [Set.mem_ofPred_eq] at hω ⊢
    obtain ⟨hL0, hall, -⟩ := hL ω
    have h8 := hall 0
    have hsq : |b 0 ω| ^ 2 < 8 / l := by
      have := pow_lt_pow_left₀ hω (abs_nonneg _) (two_ne_zero)
      rwa [Real.sq_sqrt (by positivity)] at this
    rw [lt_div_iff₀ hl] at hsq
    by_contra h
    push Not at h
    nlinarith [sq_nonneg (b 0 ω)]
  · have hsub : {ω | l < L ω} ⊆ ⋃ k, {ω | |b k ω| < √(8 / l)} := by
      intro ω hω
      obtain ⟨hL0, -, k, hk⟩ := hL ω
      simp only [Set.mem_ofPred_eq, Set.mem_iUnion] at hω ⊢
      refine ⟨k, ?_⟩
      rw [← Real.sqrt_sq (abs_nonneg (b k ω))]
      refine Real.sqrt_lt_sqrt (sq_nonneg _) ?_
      rw [lt_div_iff₀ hl]
      nlinarith [sq_nonneg (b k ω)]
    calc P.real {ω | l < L ω} ≤ P.real (⋃ k, {ω | |b k ω| < √(8 / l)}) := measureReal_mono hsub
      _ ≤ ∑ k, P.real {ω | |b k ω| < √(8 / l)} := measureReal_iUnion_fintype_le _
      _ ≤ ∑ _k : Fin 15, C * √(8 / l) := Finset.sum_le_sum fun k _ => hsb k _ hpos
      _ = 15 * C * √(8 / l) := by simp; ring

end Capri
