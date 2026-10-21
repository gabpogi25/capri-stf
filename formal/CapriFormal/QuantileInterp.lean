/-
Proposition 6 (size bound), for the quantile the statistic actually computes: numpy's linearly
interpolated p-quantile. With h = (T - 1) p, lo = ⌊h⌋ and hi = ⌈h⌉, the interpolated quantile
lies between the sorted values at lo and hi, so it is at most the k-th order statistic with
k = ⌈(T - 1) p⌉ + 1 = hi + 1. Hence {Q > c} ⊆ {fewer than k values are ≤ c}, and the bounds of
`size_bound` apply to Q itself.
-/
import Mathlib.Data.Fin.Tuple.Sort
import CapriFormal.Quantile

open MeasureTheory Finset
open scoped ENNReal

namespace Capri

variable {T : ℕ}

/-- Linear interpolation between the sorted values at `lo` and `hi`, at position `h`. -/
noncomputable def interp (s : Fin T → ℝ) (h : ℝ) (lo hi : Fin T) : ℝ :=
  s lo + (h - lo) * (s hi - s lo)

/-- The interpolated value is at most the sorted value at `hi`. -/
theorem interp_le (s : Fin T → ℝ) (hs : Monotone s) (h : ℝ) (lo hi : Fin T)
    (h1 : h - lo ≤ 1) (hle : lo ≤ hi) : interp s h lo hi ≤ s hi := by
  unfold interp
  have hmono : s lo ≤ s hi := hs hle
  nlinarith

/-- numpy's indices: for `0 ≤ p ≤ 1` and `T ≥ 1`, `⌊(T-1)p⌋` and `⌈(T-1)p⌉` are valid positions
and satisfy the hypotheses of `interp_le`. -/
theorem numpy_indices (hT : 0 < T) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) :
    ⌊((T : ℝ) - 1) * p⌋₊ < T ∧ ⌈((T : ℝ) - 1) * p⌉₊ < T ∧
    ((⌊((T : ℝ) - 1) * p⌋₊ : ℕ) : ℝ) ≤ ((T : ℝ) - 1) * p ∧
    ((T : ℝ) - 1) * p - (⌊((T : ℝ) - 1) * p⌋₊ : ℕ) ≤ 1 ∧
    ⌊((T : ℝ) - 1) * p⌋₊ ≤ ⌈((T : ℝ) - 1) * p⌉₊ := by
  have hT1 : (1 : ℝ) ≤ T := by exact_mod_cast hT
  have hh0 : 0 ≤ ((T : ℝ) - 1) * p := mul_nonneg (by linarith) hp0
  have hhT : ((T : ℝ) - 1) * p ≤ (T : ℝ) - 1 := by nlinarith
  have hceil : ⌈((T : ℝ) - 1) * p⌉₊ ≤ T - 1 := by
    rw [Nat.ceil_le]; push_cast [Nat.cast_sub hT]; linarith
  refine ⟨?_, by omega, Nat.floor_le hh0, ?_, Nat.floor_le_ceil _⟩
  · have := Nat.floor_le_ceil (((T : ℝ) - 1) * p); omega
  · have := Nat.lt_floor_add_one (((T : ℝ) - 1) * p); linarith

/-- If the interpolated quantile of the sorted values exceeds `c`, fewer than `hi + 1` of the
values are `≤ c`. Sorting is `Tuple.sort`, so this holds for any `f`. -/
theorem card_le_of_interp_gt (f : Fin T → ℝ) (c h : ℝ) (lo hi : Fin T)
    (h1 : h - lo ≤ 1) (hle : lo ≤ hi)
    (hc : c < interp (f ∘ Tuple.sort f) h lo hi) :
    (univ.filter (fun t => f t ≤ c)).card < (hi : ℕ) + 1 := by
  set σ := Tuple.sort f
  have hs : Monotone (f ∘ σ) := Tuple.monotone_sort f
  have hhi : c < f (σ hi) := lt_of_lt_of_le hc (interp_le _ hs h lo hi h1 hle)
  -- the values ≤ c sit strictly before position hi in sorted order
  have hsub : univ.filter (fun j => f (σ j) ≤ c) ⊆ Iio hi := by
    intro j hj
    simp only [mem_filter, mem_univ, true_and] at hj
    simp only [mem_Iio]
    by_contra hge
    push Not at hge
    have := hs hge
    simp only [Function.comp] at this
    linarith
  have hcard : (univ.filter (fun t => f t ≤ c)).card = (univ.filter (fun j => f (σ j) ≤ c)).card := by
    apply Finset.card_bij (fun t _ => σ.symm t)
    · intro t ht; simpa using ht
    · intro a _ b _ hab; exact σ.symm.injective hab
    · intro j hj; exact ⟨σ j, by simpa using hj, by simp⟩
  have := card_le_card hsub
  simp only [Fin.card_Iio] at this
  omega

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)

/-- **Proposition 6 for the interpolated quantile.** For any random variables `X_1..X_T` (any
joint law), with `Q` numpy's interpolated `p`-quantile and `k = ⌈(T-1)p⌉ + 1`, and any set `R` of
`k` indices: `μ {Q > c} ≤ ∑_{t ∈ R} μ {X_t > c}` and `(T - k + 1) μ {Q > c} ≤ ∑_t μ {X_t > c}`. -/
theorem size_bound_interp (X : Fin T → Ω → ℝ) (hX : ∀ t, Measurable (X t)) (c : ℝ)
    (hT : 0 < T) {p : ℝ} (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (R : Finset (Fin T))
    (hR : R.card = ⌈((T : ℝ) - 1) * p⌉₊ + 1) :
    let lo : Fin T := ⟨_, (numpy_indices hT hp0 hp1).1⟩
    let hi : Fin T := ⟨_, (numpy_indices hT hp0 hp1).2.1⟩
    let Q : Ω → ℝ := fun ω => interp ((fun t => X t ω) ∘ Tuple.sort (fun t => X t ω))
      (((T : ℝ) - 1) * p) lo hi
    μ {ω | c < Q ω} ≤ ∑ t ∈ R, μ {ω | c < X t ω} ∧
    ((T - (⌈((T : ℝ) - 1) * p⌉₊ + 1) + 1 : ℕ) : ℝ≥0∞) * μ {ω | c < Q ω}
      ≤ ∑ t, μ {ω | c < X t ω} := by
  intro lo hi Q
  obtain ⟨_, hiT, -, h1, hle⟩ := numpy_indices hT hp0 hp1
  have hsub : {ω | c < Q ω} ⊆
      {ω | (univ.filter (fun t => X t ω ≤ c)).card < ⌈((T : ℝ) - 1) * p⌉₊ + 1} := by
    intro ω hω
    exact card_le_of_interp_gt (fun t => X t ω) c _ lo hi h1 hle hω
  have hk : ⌈((T : ℝ) - 1) * p⌉₊ + 1 ≤ T := hiT
  obtain ⟨b1, b2⟩ := size_bound μ X hX c hk R hR
  exact ⟨le_trans (measure_mono hsub) b1,
    le_trans (mul_le_mul_right (measure_mono hsub) _) b2⟩

end Capri
