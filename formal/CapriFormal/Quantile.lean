/-
MATHEMATICS.md Proposition 4.5: a size bound for a low order statistic under any dependence.
The event "the k-th smallest of X_1..X_T exceeds c" is written as
"fewer than k of the X_t are at most c", which is its definition for order statistics.
Formalized: the counting facts and the two probability bounds (union bound and Markov).
Not formalized: that numpy's interpolated quantile at level p is at most the k-th order
statistic with k = ceil((T - 1) p) + 1.
-/
import Mathlib.MeasureTheory.Integral.Lebesgue.Markov
import Mathlib.Tactic.Linarith

open MeasureTheory Finset
open scoped ENNReal

namespace Capri

variable {T : ℕ}

/-- Counting core: if fewer than `k` values are `≤ c`, then at least `T - k + 1` exceed `c`. -/
theorem many_above (f : Fin T → ℝ) (c : ℝ) {k : ℕ} (hk : k ≤ T)
    (h : (univ.filter (fun t => f t ≤ c)).card < k) :
    T - k + 1 ≤ (univ.filter (fun t => c < f t)).card := by
  have hsplit := Finset.card_filter_add_card_filter_not (s := (univ : Finset (Fin T)))
    (fun t => f t ≤ c)
  simp only [card_univ, Fintype.card_fin, not_le] at hsplit
  omega

/-- Counting core: every set of `k` indices contains a value above `c`. -/
theorem exists_above (f : Fin T → ℝ) (c : ℝ) {k : ℕ} (R : Finset (Fin T)) (hR : R.card = k)
    (h : (univ.filter (fun t => f t ≤ c)).card < k) : ∃ t ∈ R, c < f t := by
  by_contra hne
  push_neg at hne
  have hsub : R ⊆ univ.filter (fun t => f t ≤ c) := fun t ht => by simp [hne t ht]
  have := card_le_card hsub
  omega

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

/-- **Proposition 4.5.** For any random variables `X_1..X_T` (any joint law), with
`E = {fewer than k of the X_t are ≤ c}`, `1 ≤ k ≤ T` and any set `R` of `k` indices:
`μ E ≤ ∑_{t ∈ R} μ {X_t > c}` and `(T - k + 1) μ E ≤ ∑_t μ {X_t > c}`. -/
theorem size_bound (X : Fin T → Ω → ℝ) (hX : ∀ t, Measurable (X t)) (c : ℝ) {k : ℕ}
    (hk : k ≤ T) (R : Finset (Fin T)) (hR : R.card = k) :
    μ {ω | (univ.filter (fun t => X t ω ≤ c)).card < k} ≤ ∑ t ∈ R, μ {ω | c < X t ω} ∧
    ((T - k + 1 : ℕ) : ℝ≥0∞) * μ {ω | (univ.filter (fun t => X t ω ≤ c)).card < k}
      ≤ ∑ t, μ {ω | c < X t ω} := by
  have hmeas : ∀ t, MeasurableSet {ω | c < X t ω} := fun t => measurableSet_lt measurable_const (hX t)
  constructor
  · calc μ {ω | (univ.filter (fun t => X t ω ≤ c)).card < k}
        ≤ μ (⋃ t ∈ R, {ω | c < X t ω}) := by
          apply measure_mono
          intro ω hω
          obtain ⟨t, ht, hlt⟩ := exists_above (fun t => X t ω) c R hR hω
          exact Set.mem_biUnion ht hlt
      _ ≤ ∑ t ∈ R, μ {ω | c < X t ω} := measure_biUnion_finset_le R _
  · -- N ω = number of indices with X_t ω > c, as an extended nonnegative real
    set N : Ω → ℝ≥0∞ := fun ω => ∑ t, Set.indicator {ω | c < X t ω} 1 ω with hN
    have hNmeas : Measurable N :=
      Finset.measurable_sum _ fun t _ => measurable_one.indicator (hmeas t)
    have hNcard : ∀ ω, N ω = ((univ.filter (fun t => c < X t ω)).card : ℝ≥0∞) := by
      intro ω
      simp only [hN, Set.indicator_apply, Set.mem_setOf_eq, Pi.one_apply]
      rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
    have hsub : {ω | (univ.filter (fun t => X t ω ≤ c)).card < k}
        ⊆ {ω | ((T - k + 1 : ℕ) : ℝ≥0∞) ≤ N ω} := by
      intro ω hω
      simp only [Set.mem_setOf_eq, hNcard]
      exact_mod_cast many_above (fun t => X t ω) c hk hω
    have hint : ∫⁻ ω, N ω ∂μ = ∑ t, μ {ω | c < X t ω} := by
      rw [lintegral_finsetSum _ fun t _ => measurable_one.indicator (hmeas t)]
      refine Finset.sum_congr rfl fun t _ => ?_
      rw [lintegral_indicator_one (hmeas t)]
    calc ((T - k + 1 : ℕ) : ℝ≥0∞) * μ {ω | (univ.filter (fun t => X t ω ≤ c)).card < k}
        ≤ ((T - k + 1 : ℕ) : ℝ≥0∞) * μ {ω | ((T - k + 1 : ℕ) : ℝ≥0∞) ≤ N ω} :=
          mul_le_mul_right (measure_mono hsub) _
      _ ≤ ∫⁻ ω, N ω ∂μ := mul_meas_ge_le_lintegral hNmeas _
      _ = ∑ t, μ {ω | c < X t ω} := hint

end Capri
