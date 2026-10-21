/-
Proposition 5.1 (first-order noise law): with isotropic Gaussian noise `σX`, `X` standard
Gaussian on the coordinate space, and first-order errors `Y_k = σ⟪g_k, X⟫` of the three
invariants (`g_k` their gradients, κ = max_k ‖g_k‖), the median of `M = max_k |Y_k|` lies
between `Φ⁻¹(3/4) σ κ` and `Φ⁻¹(11/12) σ κ`.

The medians are stated as `P(M < Φ⁻¹(3/4) σ κ) ≤ 1/2` and `P(M > Φ⁻¹(11/12) σ κ) ≤ 1/2`,
with `Φ⁻¹(3/4)` and `Φ⁻¹(11/12)` any numbers where the standard normal distribution function
takes those values. (Their decimal values 0.674 and 1.383 are not part of the formal statement.)
-/
import CapriFormal.LimitLaw

open MeasureTheory ProbabilityTheory
open scoped RealInnerProductSpace

namespace Capri

section Normal

/-- Symmetry of the standard normal law. -/
theorem gaussian_symm (s : Set ℝ) (hs : MeasurableSet s) :
    (gaussianReal 0 1).real ((fun x => -x) ⁻¹' s) = (gaussianReal 0 1).real s := by
  have h := gaussianReal_map_neg (μ := 0) (v := 1)
  rw [neg_zero] at h
  rw [measureReal_def, measureReal_def, ← Measure.map_apply measurable_neg hs, h]

theorem gaussian_lt_neg (t : ℝ) :
    (gaussianReal 0 1).real {x | x < -t} = (gaussianReal 0 1).real {x | t < x} := by
  have := gaussian_symm {x | t < x} (measurableSet_lt measurable_const measurable_id)
  rw [← this]
  congr 1
  ext x; simp [lt_neg]

theorem gaussian_le_neg (t : ℝ) :
    (gaussianReal 0 1).real {x | x ≤ -t} = (gaussianReal 0 1).real {x | t ≤ x} := by
  have := gaussian_symm {x | t ≤ x} (measurableSet_le measurable_const measurable_id)
  rw [← this]
  congr 1
  ext x; simp [le_neg]

local notation "𝒩" => gaussianReal 0 1

theorem nreal_union {s t : Set ℝ} (hd : Disjoint s t) (ht : MeasurableSet t) :
    (𝒩).real s + (𝒩).real t = (𝒩).real (s ∪ t) :=
  (measureReal_union hd ht).symm

theorem le_add_gt (c : ℝ) : (𝒩).real {x | x ≤ c} + (𝒩).real {x | c < x} = 1 := by
  rw [nreal_union (s := {x | x ≤ c}) (t := {x | c < x})
    (Set.disjoint_left.2 fun x (hx : x ≤ c) (hx' : c < x) => by linarith)
    (measurableSet_lt measurable_const measurable_id)]
  have : {x : ℝ | x ≤ c} ∪ {x | c < x} = Set.univ := by
    ext x; simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_univ, iff_true]; exact le_or_gt x c
  rw [this]; simp

theorem ge_add_lt (c : ℝ) : (𝒩).real {x | c ≤ x} + (𝒩).real {x | x < c} = 1 := by
  rw [nreal_union (s := {x | c ≤ x}) (t := {x | x < c})
    (Set.disjoint_left.2 fun x (hx : c ≤ x) (hx' : x < c) => by linarith)
    (measurableSet_lt measurable_id measurable_const)]
  have : {x : ℝ | c ≤ x} ∪ {x | x < c} = Set.univ := by
    ext x; simp only [Set.mem_union, Set.mem_setOf_eq, Set.mem_univ, iff_true]; exact le_or_gt c x
  rw [this]; simp

/-- A point where the standard normal distribution function exceeds 1/2 is nonnegative. -/
theorem nonneg_of_cdf {c : ℝ} (hc : 1 / 2 < (𝒩).real {x | x ≤ c}) : 0 ≤ c := by
  by_contra h
  push Not at h
  have h1 : (𝒩).real {x | x ≤ c} ≤ (𝒩).real {x | x < -c} :=
    measureReal_mono fun x (hx : x ≤ c) => show x < -c by linarith
  have h2 := le_add_gt c
  have h3 : (𝒩).real {x | x < -c} = (𝒩).real {x | c < x} := gaussian_lt_neg c
  linarith

/-- With `F(a) = 3/4`: `P(|N| < a) ≤ 1/2`. -/
theorem gaussian_abs_lt {a : ℝ} (ha : (𝒩).real {x | x ≤ a} = 3 / 4) :
    (𝒩).real {x | |x| < a} ≤ 1 / 2 := by
  have ha0 : 0 ≤ a := nonneg_of_cdf (by rw [ha]; norm_num)
  have h1 : (𝒩).real {x | |x| < a} + (𝒩).real {x | x ≤ -a} ≤ (𝒩).real {x | x ≤ a} := by
    rw [nreal_union (s := {x | |x| < a}) (t := {x | x ≤ -a})
      (Set.disjoint_left.2 fun x (hx : |x| < a) (hx' : x ≤ -a) => by
        have := (abs_lt.1 hx).1; linarith)
      (measurableSet_le measurable_id measurable_const)]
    refine measureReal_mono fun x hx => ?_
    show x ≤ a
    rcases hx with hx | hx
    · exact le_of_lt (abs_lt.1 (show |x| < a from hx)).2
    · have hx' : x ≤ -a := hx
      linarith
  have h2 := ge_add_lt a
  have h3 : (𝒩).real {x | x < a} ≤ (𝒩).real {x | x ≤ a} :=
    measureReal_mono fun x (hx : x < a) => show x ≤ a from le_of_lt hx
  rw [gaussian_le_neg a] at h1
  linarith

/-- With `F(b) = 11/12`: `P(|N| > b) ≤ 1/6`. -/
theorem gaussian_abs_gt {b : ℝ} (hb : (𝒩).real {x | x ≤ b} = 11 / 12) :
    (𝒩).real {x | b < |x|} ≤ 1 / 6 := by
  have hcompl : (𝒩).real {x | b < x} = 1 / 12 := by
    have := le_add_gt b; linarith
  have hsub : {x : ℝ | b < |x|} ⊆ {x | b < x} ∪ {x | x < -b} := by
    intro x hx
    have hx' : b < |x| := hx
    rcases abs_cases x with ⟨h, _⟩ | ⟨h, _⟩
    · left; show b < x; linarith
    · right; show x < -b; linarith
  calc (𝒩).real {x | b < |x|} ≤ (𝒩).real ({x | b < x} ∪ {x | x < -b}) := measureReal_mono hsub
    _ ≤ (𝒩).real {x | b < x} + (𝒩).real {x | x < -b} := measureReal_union_le _ _
    _ = 1 / 6 := by rw [gaussian_lt_neg, hcompl]; norm_num

/-- `Φ⁻¹(11/12) ≥ 0`. -/
theorem quantile_nonneg {b : ℝ} (hb : (𝒩).real {x | x ≤ b} = 11 / 12) : 0 ≤ b :=
  nonneg_of_cdf (by rw [hb]; norm_num)

end Normal

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- The law of a normalised projection, as probabilities of sets. -/
theorem proj_real {g : E} (hg : g ≠ 0) (s : Set ℝ) (hs : MeasurableSet s) :
    (stdGaussian E).real ((fun x => ⟪g, x⟫ / ‖g‖) ⁻¹' s) = (gaussianReal 0 1).real s := by
  rw [measureReal_def, measureReal_def, ← Measure.map_apply (by fun_prop) hs,
    stdGaussian_map_inner_div hg]

/-- **Proposition 5.1 (first-order noise law).** Let `X` be standard Gaussian, `σ > 0`, `g_k`
the gradients of the three invariants and `κ = max_k ‖g_k‖ > 0`, and
`M = max_k |σ⟪g_k, X⟫|`. If `F(a) = 3/4` and `F(b) = 11/12` (F the standard normal
distribution function), then `P(M < aσκ) ≤ 1/2 ≤ P(M ≤ bσκ)`: the median of `M` lies in
`[aσκ, bσκ]`. -/
theorem noise_law (g : Fin 3 → E) {σ : ℝ} (hσ : 0 < σ)
    (hκ : 0 < Finset.univ.sup' Finset.univ_nonempty fun k => ‖g k‖)
    {a b : ℝ} (ha : (gaussianReal 0 1).real {x | x ≤ a} = 3 / 4)
    (hb : (gaussianReal 0 1).real {x | x ≤ b} = 11 / 12) :
    let κ := Finset.univ.sup' Finset.univ_nonempty fun k => ‖g k‖
    let M : E → ℝ := fun x => Finset.univ.sup' Finset.univ_nonempty fun k => |σ * ⟪g k, x⟫|
    (stdGaussian E).real {x | M x < a * σ * κ} ≤ 1 / 2 ∧
    (stdGaussian E).real {x | b * σ * κ < M x} ≤ 1 / 2 := by
  classical
  intro κ M
  obtain ⟨k0, -, hk0⟩ := Finset.exists_max_image Finset.univ (fun k => ‖g k‖) Finset.univ_nonempty
  have hκk0 : κ = ‖g k0‖ :=
    le_antisymm (Finset.sup'_le _ _ fun k _ => hk0 k (Finset.mem_univ k))
      (Finset.le_sup' (fun k => ‖g k‖) (Finset.mem_univ k0))
  have hg0 : g k0 ≠ 0 := by
    intro h
    have hpos : 0 < κ := hκ
    rw [hκk0, h, norm_zero] at hpos; exact lt_irrefl 0 hpos
  have hn0 : 0 < ‖g k0‖ := norm_pos_iff.2 hg0
  constructor
  · -- lower bound: M ≥ |Y_{k0}|
    have hsub : {x | M x < a * σ * κ} ⊆ (fun x => ⟪g k0, x⟫ / ‖g k0‖) ⁻¹' {y | |y| < a} := by
      intro x hx
      simp only [Set.mem_ofPred_eq, Set.mem_preimage] at hx ⊢
      have hle : |σ * ⟪g k0, x⟫| ≤ M x :=
        Finset.le_sup' (fun k => |σ * ⟪g k, x⟫|) (Finset.mem_univ k0)
      rw [abs_div, abs_norm, div_lt_iff₀ hn0]
      rw [abs_mul, abs_of_pos hσ, hκk0] at *
      nlinarith
    calc (stdGaussian E).real {x | M x < a * σ * κ}
        ≤ (stdGaussian E).real ((fun x => ⟪g k0, x⟫ / ‖g k0‖) ⁻¹' {y | |y| < a}) :=
          measureReal_mono hsub
      _ = (gaussianReal 0 1).real {y | |y| < a} :=
          proj_real hg0 _ (measurableSet_lt (by fun_prop) measurable_const)
      _ ≤ 1 / 2 := gaussian_abs_lt ha
  · -- upper bound: union bound over the three invariants
    have hb0 := quantile_nonneg hb
    have hsub : {x | b * σ * κ < M x} ⊆
        ⋃ k ∈ Finset.univ.filter (fun k => g k ≠ 0),
          (fun x => ⟪g k, x⟫ / ‖g k‖) ⁻¹' {y | b < |y|} := by
      intro x hx
      simp only [Set.mem_ofPred_eq] at hx
      obtain ⟨k, -, hk⟩ := Finset.exists_max_image Finset.univ
        (fun k => |σ * ⟪g k, x⟫|) Finset.univ_nonempty
      have hMk : M x = |σ * ⟪g k, x⟫| :=
        le_antisymm (Finset.sup'_le _ _ fun k' _ => hk k' (Finset.mem_univ k'))
          (Finset.le_sup' (fun k => |σ * ⟪g k, x⟫|) (Finset.mem_univ k))
      have hgk : g k ≠ 0 := by
        intro h
        rw [hMk, h, inner_zero_left, mul_zero, abs_zero] at hx
        have : 0 ≤ b * σ * κ := by positivity
        linarith
      have hnk : 0 < ‖g k‖ := norm_pos_iff.2 hgk
      have hκk : ‖g k‖ ≤ κ := Finset.le_sup' (fun k => ‖g k‖) (Finset.mem_univ k)
      simp only [Set.mem_iUnion, Set.mem_preimage, Set.mem_ofPred_eq, Finset.mem_filter,
        Finset.mem_univ, true_and]
      refine ⟨k, hgk, ?_⟩
      rw [abs_div, abs_norm, lt_div_iff₀ hnk]
      rw [hMk, abs_mul, abs_of_pos hσ] at hx
      have : b * σ * ‖g k‖ ≤ b * σ * κ := by
        have := mul_le_mul_of_nonneg_left hκk (mul_nonneg hb0 hσ.le); linarith
      nlinarith
    calc (stdGaussian E).real {x | b * σ * κ < M x}
        ≤ (stdGaussian E).real (⋃ k ∈ Finset.univ.filter (fun k => g k ≠ 0),
            (fun x => ⟪g k, x⟫ / ‖g k‖) ⁻¹' {y | b < |y|}) := measureReal_mono hsub (measure_ne_top _ _)
      _ ≤ ∑ k ∈ Finset.univ.filter (fun k => g k ≠ 0),
            (stdGaussian E).real ((fun x => ⟪g k, x⟫ / ‖g k‖) ⁻¹' {y | b < |y|}) :=
          measureReal_biUnion_finset_le _ _
      _ ≤ ∑ _k ∈ Finset.univ.filter (fun k : Fin 3 => g k ≠ 0), (1 / 6 : ℝ) := by
          refine Finset.sum_le_sum fun k hk => ?_
          rw [proj_real (Finset.mem_filter.1 hk).2 _
            (measurableSet_lt measurable_const (by fun_prop))]
          exact gaussian_abs_gt hb
      _ ≤ ∑ _k : Fin 3, (1 / 6 : ℝ) := Finset.sum_le_sum_of_subset_of_nonneg
          (Finset.filter_subset _ _) fun _ _ _ => by norm_num
      _ = 1 / 2 := by simp; norm_num

end Capri
