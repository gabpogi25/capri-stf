/-
The combinatorial and algebraic parts of Proposition 10 (MATHEMATICS.md Proposition 5.5, the rate of
κ_J near a boundary stratum). Points are labelled 0..5 (1..6 in the text).

* `used_card`, `unused_eq`: J1-J3 use 11 of the 15 brackets; the other four are
  [1236], [1245], [1456], [2356].
* `axis_points`: each J_k has two axis points, in all four of its brackets, and its four other
  points each lie in exactly two of them.
* `inv_entry`: an entry of `M⁻¹` is a 3×3 cofactor over `det M`, so a row of `M⁻¹` grows like
  `1/|det M|` when the cofactor stays away from 0.
* `rate_upper`, `rate_lower`: the inequalities that turn `c' d ≤ |b₀| ≤ C'' d` into
  `κ_J ≤ 8/(c' d)` and `‖row₀ - row₁‖ ≥ c/d` for small `d`.

Not formalized: the smoothness of `d ↦ b₀(P_d)` and the gradient formula
`∂ log|det M| / ∂M = M^{-T}` (Jacobi's formula), which supply the hypotheses of `rate_lower`.
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace Capri

/-- The brackets of J1 = [1234][1256]/([1235][1246]). -/
def J1 : Finset (Finset (Fin 6)) := {{0, 1, 2, 3}, {0, 1, 4, 5}, {0, 1, 2, 4}, {0, 1, 3, 5}}
/-- The brackets of J2 = [1345][2346]/([1346][2345]). -/
def J2 : Finset (Finset (Fin 6)) := {{0, 2, 3, 4}, {1, 2, 3, 5}, {0, 2, 3, 5}, {1, 2, 3, 4}}
/-- The brackets of J3 = [1256][3456]/([1356][2456]). -/
def J3 : Finset (Finset (Fin 6)) := {{0, 1, 4, 5}, {2, 3, 4, 5}, {0, 2, 4, 5}, {1, 3, 4, 5}}

/-- All 15 brackets of six points. -/
def allBrackets : Finset (Finset (Fin 6)) := Finset.univ.powersetCard 4

/-- The brackets used by J1-J3. -/
def used : Finset (Finset (Fin 6)) := J1 ∪ J2 ∪ J3

theorem all_card : allBrackets.card = 15 := by decide

theorem used_card : used.card = 11 := by decide

theorem used_sub : used ⊆ allBrackets := by decide

/-- The four brackets J1-J3 do not use: [1236], [1245], [1456], [2356]. -/
theorem unused_eq :
    allBrackets \ used = {{0, 1, 2, 5}, {0, 1, 3, 4}, {0, 3, 4, 5}, {1, 2, 4, 5}} := by decide

/-- Each J_k has two axis points that lie in all four of its brackets; each other point lies in
exactly two. -/
theorem axis_points :
    (∀ B ∈ J1, (0 : Fin 6) ∈ B ∧ (1 : Fin 6) ∈ B) ∧
    (∀ x : Fin 6, x ≠ 0 → x ≠ 1 → (J1.filter (x ∈ ·)).card = 2) ∧
    (∀ B ∈ J2, (2 : Fin 6) ∈ B ∧ (3 : Fin 6) ∈ B) ∧
    (∀ x : Fin 6, x ≠ 2 → x ≠ 3 → (J2.filter (x ∈ ·)).card = 2) ∧
    (∀ B ∈ J3, (4 : Fin 6) ∈ B ∧ (5 : Fin 6) ∈ B) ∧
    (∀ x : Fin 6, x ≠ 4 → x ≠ 5 → (J3.filter (x ∈ ·)).card = 2) := by decide

/-- An entry of the inverse is a cofactor over the determinant:
`|M⁻¹ i j| = |adj M i j| / |det M|`. -/
theorem inv_entry {n : Type*} [Fintype n] [DecidableEq n] (M : Matrix n n ℝ) (i j : n) :
    |M⁻¹ i j| = |M.adjugate i j| / |M.det| := by
  rw [Matrix.inv_def, Ring.inverse_eq_inv', Matrix.smul_apply, smul_eq_mul, abs_mul, abs_inv,
    div_eq_inv_mul]

/-- **Upper rate.** If `κ_J ≤ 8/δ_J`, `δ_J = |b₀|` and `|b₀| ≥ c' d`, then `κ_J ≤ 8/(c' d)`. -/
theorem rate_upper {κ b₀ c' d : ℝ} (hc : 0 < c') (hd : 0 < d) (hκ : κ ≤ 8 / |b₀|)
    (hb : c' * d ≤ |b₀|) : κ ≤ 8 / (c' * d) :=
  le_trans hκ (div_le_div_of_nonneg_left (by norm_num) (mul_pos hc hd) hb)

/-- **Lower rate.** If one term of the gradient has norm at least `m/|b₀|` with `|b₀| ≤ C d`,
and the other at most `K`, their difference has norm at least `m/(2 C d)` once
`d ≤ m/(2 C K)`. -/
theorem rate_lower {A B b₀ m C K d : ℝ} (hm : 0 < m) (hC : 0 < C) (hK : 0 < K) (hd : 0 < d)
    (hb0 : 0 < |b₀|) (hA : m / |b₀| ≤ A) (hb : |b₀| ≤ C * d) (hB : B ≤ K)
    (hsmall : d ≤ m / (2 * C * K)) : m / (2 * C * d) ≤ A - B := by
  have h1 : m / (C * d) ≤ A :=
    le_trans (div_le_div_of_nonneg_left hm.le hb0 hb) hA
  have h2 : K ≤ m / (2 * C * d) := by
    rw [le_div_iff₀ (by positivity)]
    rw [le_div_iff₀ (by positivity)] at hsmall
    nlinarith
  have h3 : m / (C * d) = 2 * (m / (2 * C * d)) := by field_simp
  linarith

end Capri
