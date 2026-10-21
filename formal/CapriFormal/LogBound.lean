/-
The bound of MATHEMATICS.md Theorem 5.3, from the bracket perturbation onwards.
Formalized: if every bracket moves by at most `ε < |b|`, then
`|Δ log|J|| ≤ ∑ -log(1 - ε/|b_i|)`, and the uniform bound with the smallest bracket.
Not formalized: the step `|b(a) - b(q)| ≤ (1 + δ)^4 - 1` (multilinear expansion and
Hadamard's inequality).
-/
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace Capri

open Real

/-- One bracket: `|log|b'| - log|b|| ≤ -log(1 - ε/|b|)` when `|b' - b| ≤ ε < |b|`. -/
theorem log_abs_change {b b' ε : ℝ} (hε : ε < |b|) (h : |b' - b| ≤ ε) :
    abs (log |b'| - log |b|) ≤ -log (1 - ε / |b|) := by
  have hb : 0 < |b| := lt_of_le_of_lt (le_trans (abs_nonneg _) h) hε
  have hε0 : 0 ≤ ε := le_trans (abs_nonneg _) h
  set x := ε / |b| with hxdef
  have hx0 : 0 ≤ x := div_nonneg hε0 hb.le
  have hx1 : x < 1 := (div_lt_one hb).mpr hε
  have hd : |(|b'| - |b|)| ≤ ε := le_trans (abs_abs_sub_abs_le_abs_sub b' b) h
  have hlo : |b| * (1 - x) ≤ |b'| := by
    have : |b| * (1 - x) = |b| - ε := by rw [hxdef]; field_simp
    rw [this]; linarith [(abs_le.mp hd).1]
  have hhi : |b'| ≤ |b| * (1 + x) := by
    have : |b| * (1 + x) = |b| + ε := by rw [hxdef]; field_simp
    rw [this]; linarith [(abs_le.mp hd).2]
  have h1x : 0 < 1 - x := by linarith
  have hb' : 0 < |b'| := lt_of_lt_of_le (mul_pos hb h1x) hlo
  have hdiff : log |b'| - log |b| = log (|b'| / |b|) := (log_div hb'.ne' hb.ne').symm
  have hr_lo : 1 - x ≤ |b'| / |b| := by rw [le_div_iff₀ hb]; linarith
  have hr_hi : |b'| / |b| ≤ 1 + x := by rw [div_le_iff₀ hb]; linarith
  have hr : 0 < |b'| / |b| := div_pos hb' hb
  -- log(1 + x) ≤ -log(1 - x) because (1 + x)(1 - x) ≤ 1
  have hplus : log (1 + x) ≤ -log (1 - x) := by
    have hprod : log ((1 + x) * (1 - x)) ≤ 0 :=
      log_nonpos (by nlinarith) (by nlinarith)
    rw [log_mul (by linarith) h1x.ne'] at hprod
    linarith
  rw [hdiff, abs_le]
  constructor
  · have := log_le_log h1x hr_lo
    linarith
  · exact le_trans (log_le_log hr hr_hi) hplus

/-- Monotonicity in the bracket: a smaller bracket gives a larger bound. -/
theorem neg_log_mono {ε d B : ℝ} (hε0 : 0 ≤ ε) (hεd : ε < d) (hdB : d ≤ B) :
    -log (1 - ε / B) ≤ -log (1 - ε / d) := by
  have hd : 0 < d := lt_of_le_of_lt hε0 hεd
  have hB : 0 < B := lt_of_lt_of_le hd hdB
  have h1 : ε / B ≤ ε / d := div_le_div_of_nonneg_left hε0 hd hdB
  have h2 : ε / d < 1 := (div_lt_one hd).mpr hεd
  have := log_le_log (by linarith) (by linarith : 1 - ε / d ≤ 1 - ε / B)
  linarith

/-- **Theorem 5.3, from the brackets on.** `log|J| = log|b1| + log|b2| - log|b3| - log|b4|`.
If each bracket moves by at most `ε < |b_i|`, the change of `log|J|` is at most the sum of the
four terms; if moreover `ε < δ ≤ |b_i|` for all `i`, it is at most `-4 log(1 - ε/δ)`. -/
theorem logJ_bound (b b' : Fin 4 → ℝ) (ε δ : ℝ) (hε : ∀ i, ε < |b i|) (h : ∀ i, |b' i - b i| ≤ ε) :
    abs ((log |b' 0| + log |b' 1| - log |b' 2| - log |b' 3|) -
        (log |b 0| + log |b 1| - log |b 2| - log |b 3|))
      ≤ ∑ i, -log (1 - ε / |b i|) ∧
    (ε < δ → (∀ i, δ ≤ |b i|) →
      abs ((log |b' 0| + log |b' 1| - log |b' 2| - log |b' 3|) -
          (log |b 0| + log |b 1| - log |b 2| - log |b 3|)) ≤ -4 * log (1 - ε / δ)) := by
  have hb := fun i => abs_le.mp (log_abs_change (hε i) (h i))
  have first : abs ((log |b' 0| + log |b' 1| - log |b' 2| - log |b' 3|) -
        (log |b 0| + log |b 1| - log |b 2| - log |b 3|)) ≤ ∑ i, -log (1 - ε / |b i|) := by
    rw [Fin.sum_univ_four, abs_le]
    have := hb 0; have := hb 1; have := hb 2; have := hb 3
    constructor <;> linarith
  refine ⟨first, fun hεδ hδ => le_trans first ?_⟩
  have hε0 : 0 ≤ ε := le_trans (abs_nonneg _) (h 0)
  rw [Fin.sum_univ_four]
  have := neg_log_mono hε0 hεδ (hδ 0); have := neg_log_mono hε0 hεδ (hδ 1)
  have := neg_log_mono hε0 hεδ (hδ 2); have := neg_log_mono hε0 hεδ (hδ 3)
  linarith

end Capri
