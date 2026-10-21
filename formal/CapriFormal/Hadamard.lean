/-
Hadamard's inequality and the bracket perturbation behind Theorems 6 and 7.

* `hadamard`: |det(v_1, ..., v_n)| ≤ ∏ ‖v_i‖ in Euclidean space (Gram–Schmidt).
* `det_perturb`: if ‖q_i‖ ≤ 1 and ‖u_i‖ ≤ δ, then |det(q + u) - det q| ≤ (1 + δ)^n - 1
  (multilinear expansion over the subsets of replaced columns, each term bounded by Hadamard).
* `br_hadamard`, `br_perturb`: the same for the bracket `br` of four vectors of ℝ⁴.
* `tau_exact`: Theorem 6. Each Plücker relation, evaluated on brackets whose representatives
  are within δ of unit representatives (possibly different ones for different brackets), is at
  most 3ε(2 + ε) in absolute value, ε = (1 + δ)^4 - 1.
* `logJ_bound_vectors`: Theorem 7 from the vectors on.
-/
import Mathlib.Analysis.InnerProductSpace.GramSchmidtOrtho
import Mathlib.Analysis.InnerProductSpace.PiL2
import CapriFormal.Pencil
import CapriFormal.LogBound

open Module Finset InnerProductSpace

namespace Capri

/-- **Hadamard's inequality.** The determinant of `n` vectors of `ℝⁿ` (in the standard basis)
is at most the product of their Euclidean norms in absolute value. -/
theorem hadamard {n : ℕ} (f : Fin n → EuclideanSpace ℝ (Fin n)) :
    |(EuclideanSpace.basisFun (Fin n) ℝ).toBasis.det f| ≤ ∏ i, ‖f i‖ := by
  classical
  have h : finrank ℝ (EuclideanSpace ℝ (Fin n)) = Fintype.card (Fin n) := by simp
  set b := gramSchmidtOrthonormalBasis h f with hb
  set e := EuclideanSpace.basisFun (Fin n) ℝ with he
  have hmul : e.toBasis.det f = e.toBasis.det b * b.toBasis.det f := by
    rw [Basis.det_apply, Basis.det_apply, Basis.det_apply, ← Matrix.det_mul,
      ← OrthonormalBasis.coe_toBasis, Basis.toMatrix_mul_toMatrix]
  have hunit : |e.toBasis.det b| = 1 := by
    have := e.det_to_matrix_orthonormalBasis b
    rwa [Real.norm_eq_abs] at this
  rw [hmul, abs_mul, hunit, one_mul, gramSchmidtOrthonormalBasis_det, Finset.abs_prod]
  refine Finset.prod_le_prod₀ (fun i _ => abs_nonneg _) (fun i _ => ?_)
  calc |inner ℝ (b i) (f i)| ≤ ‖b i‖ * ‖f i‖ := abs_real_inner_le_norm _ _
    _ = ‖f i‖ := by rw [b.orthonormal.1 i, one_mul]

/-- **Bracket perturbation.** If every `‖q_i‖ ≤ 1` and `‖u_i‖ ≤ δ`, the determinant moves by at
most `(1 + δ)^n - 1`. -/
theorem det_perturb {n : ℕ} (q u : Fin n → EuclideanSpace ℝ (Fin n)) {δ : ℝ}
    (hq : ∀ i, ‖q i‖ ≤ 1) (hu : ∀ i, ‖u i‖ ≤ δ) :
    |(EuclideanSpace.basisFun (Fin n) ℝ).toBasis.det (q + u)
        - (EuclideanSpace.basisFun (Fin n) ℝ).toBasis.det q| ≤ (1 + δ) ^ n - 1 := by
  classical
  set D := (EuclideanSpace.basisFun (Fin n) ℝ).toBasis.det with hD
  have hexp : D (u + q) = ∑ s : Finset (Fin n), D (s.piecewise u q) :=
    D.toMultilinearMap.map_add_univ u q
  have hempty : D ((∅ : Finset (Fin n)).piecewise u q) = D q := by simp
  have hsplit : ∑ s : Finset (Fin n), D (s.piecewise u q)
      = D q + ∑ s ∈ (univ : Finset (Finset (Fin n))).erase ∅, D (s.piecewise u q) := by
    rw [← Finset.add_sum_erase _ _ (mem_univ ∅), hempty]
  -- each term is bounded by Hadamard
  set g : Finset (Fin n) → ℝ := fun s => ∏ i, (if i ∈ s then δ else 1) with hg
  have hterm : ∀ s : Finset (Fin n), |D (s.piecewise u q)| ≤ g s := by
    intro s
    refine le_trans (hadamard _) (Finset.prod_le_prod₀ (fun i _ => norm_nonneg _) ?_)
    intro i _
    by_cases hi : i ∈ s <;> simp [Finset.piecewise, hi, hu i, hq i]
  have hg_eq : ∀ s : Finset (Fin n), g s = δ ^ s.card * 1 ^ (n - s.card) := by
    intro s
    simp only [hg, one_pow, mul_one]
    rw [Finset.prod_ite, Finset.prod_const, Finset.prod_const_one, mul_one,
      Finset.filter_mem_eq_inter, univ_inter]
  have htotal : ∑ s : Finset (Fin n), g s = (1 + δ) ^ n := by
    simp_rw [hg_eq]
    rw [Fin.sum_pow_mul_eq_add_pow, add_comm]
  have hg_empty : g ∅ = 1 := by simp [hg]
  have hrest : ∑ s ∈ (univ : Finset (Finset (Fin n))).erase ∅, g s = (1 + δ) ^ n - 1 := by
    rw [← htotal, ← Finset.add_sum_erase _ _ (mem_univ ∅), hg_empty]; ring
  rw [add_comm q u, hexp, hsplit, add_sub_cancel_left, ← hrest]
  exact le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun s _ => hterm s)

/-- The bracket is the determinant of the four vectors in the standard basis. -/
theorem br_eq_det (a b c d : Fin 4 → ℝ) :
    br a b c d = (EuclideanSpace.basisFun (Fin 4) ℝ).toBasis.det
      ![WithLp.toLp 2 a, WithLp.toLp 2 b, WithLp.toLp 2 c, WithLp.toLp 2 d] := by
  rw [Basis.det_apply, br, ← Matrix.det_transpose]
  congr 1
  ext i j
  fin_cases i <;> fin_cases j <;> simp [Basis.toMatrix_apply, Matrix.transpose_apply]

/-- Euclidean norm of a coordinate vector. -/
local notation "‖" x "‖₂" => ‖WithLp.toLp 2 x‖

/-- Hadamard for brackets: `|[abcd]| ≤ ‖a‖‖b‖‖c‖‖d‖` (Euclidean norms). -/
theorem br_hadamard (a b c d : Fin 4 → ℝ) : |br a b c d| ≤ ‖a‖₂ * ‖b‖₂ * ‖c‖₂ * ‖d‖₂ := by
  rw [br_eq_det]
  refine le_trans (hadamard _) (le_of_eq ?_)
  simp [Fin.prod_univ_four]

/-- Bracket perturbation in ℝ⁴: representatives within `δ` of vectors of norm at most 1 move a
bracket by at most `ε = (1 + δ)^4 - 1`. -/
theorem br_perturb (q u : Fin 4 → Fin 4 → ℝ) {δ : ℝ}
    (hq : ∀ i, ‖q i‖₂ ≤ 1) (hu : ∀ i, ‖u i‖₂ ≤ δ) :
    |br (q 0 + u 0) (q 1 + u 1) (q 2 + u 2) (q 3 + u 3) - br (q 0) (q 1) (q 2) (q 3)|
      ≤ (1 + δ) ^ 4 - 1 := by
  rw [br_eq_det, br_eq_det]
  have h1 : (![WithLp.toLp 2 (q 0 + u 0), WithLp.toLp 2 (q 1 + u 1), WithLp.toLp 2 (q 2 + u 2),
      WithLp.toLp 2 (q 3 + u 3)] : Fin 4 → EuclideanSpace ℝ (Fin 4))
      = (fun i => WithLp.toLp 2 (q i)) + (fun i => WithLp.toLp 2 (u i)) := by
    funext i; fin_cases i <;> simp [WithLp.toLp_add]
  have h2 : (![WithLp.toLp 2 (q 0), WithLp.toLp 2 (q 1), WithLp.toLp 2 (q 2), WithLp.toLp 2 (q 3)] :
      Fin 4 → EuclideanSpace ℝ (Fin 4)) = (fun i => WithLp.toLp 2 (q i)) := by
    funext i; fin_cases i <;> simp
  rw [h1, h2]
  exact det_perturb (n := 4) (fun i => WithLp.toLp 2 (q i)) (fun i => WithLp.toLp 2 (u i)) hq hu

/-- Three products of brackets with `|x_i| ≤ 1`, each bracket moved by at most `ε`, and the
signed sum zero at the truth: the measured signed sum is at most `3ε(2 + ε)`. -/
theorem three_products (x y : Fin 6 → ℝ) {ε : ℝ} (hε : 0 ≤ ε) (hx : ∀ i, |x i| ≤ 1)
    (hd : ∀ i, |y i - x i| ≤ ε) (h0 : x 0 * x 1 - x 2 * x 3 + x 4 * x 5 = 0) :
    |y 0 * y 1 - y 2 * y 3 + y 4 * y 5| ≤ 3 * (ε * (2 + ε)) := by
  have prod : ∀ i j, |y i * y j - x i * x j| ≤ ε * (2 + ε) := by
    intro i j
    have hyj : |y j| ≤ 1 + ε := by
      have := abs_sub_abs_le_abs_sub (y j) (x j); linarith [hx j, hd j]
    have e : y i * y j - x i * x j = (y i - x i) * y j + x i * (y j - x j) := by ring
    rw [e]
    calc |(y i - x i) * y j + x i * (y j - x j)|
        ≤ |y i - x i| * |y j| + |x i| * |y j - x j| := by
          rw [← abs_mul, ← abs_mul]; exact abs_add_le _ _
      _ ≤ ε * (1 + ε) + 1 * ε := by
          gcongr
          · exact hd i
          · exact hx i
          · exact hd j
      _ = ε * (2 + ε) := by ring
  have e : y 0 * y 1 - y 2 * y 3 + y 4 * y 5
      = (y 0 * y 1 - x 0 * x 1) - (y 2 * y 3 - x 2 * x 3) + (y 4 * y 5 - x 4 * x 5) := by
    linear_combination h0
  rw [e]
  calc |(y 0 * y 1 - x 0 * x 1) - (y 2 * y 3 - x 2 * x 3) + (y 4 * y 5 - x 4 * x 5)|
      ≤ |y 0 * y 1 - x 0 * x 1| + |y 2 * y 3 - x 2 * x 3| + |y 4 * y 5 - x 4 * x 5| := by
        refine le_trans (abs_add_le _ _) ?_
        gcongr
        exact abs_sub _ _
    _ ≤ 3 * (ε * (2 + ε)) := by linarith [prod 0 1, prod 2 3, prod 4 5]

/-- **Theorem 6 (exact drift bound).** Let `a, ..., f` be unit vectors. Evaluate the Plücker
relation `GP_ab = [abcd][abef] - [abce][abdf] + [abcf][abde]` with each of its six brackets
computed from its own representatives, each within `δ` of the unit vector it stands for. Then
`|GP_ab| ≤ 3ε(2 + ε)` with `ε = (1 + δ)^4 - 1`. The relations Σ₁, Σ₂, Σ₃ are such pencils
(up to sign), so this is `ρ(S) ≤ τ_exact(δ)`. -/
theorem tau_exact (a b c d e f : Fin 4 → ℝ) (ha : ‖a‖₂ = 1) (hb : ‖b‖₂ = 1) (hc : ‖c‖₂ = 1)
    (hd : ‖d‖₂ = 1) (he : ‖e‖₂ = 1) (hf : ‖f‖₂ = 1) {δ : ℝ} (hδ : 0 ≤ δ)
    (u : Fin 6 → Fin 4 → Fin 4 → ℝ) (hu : ∀ j i, ‖u j i‖₂ ≤ δ) :
    let P : Fin 6 → Fin 4 → Fin 4 → ℝ :=
      ![![a, b, c, d], ![a, b, e, f], ![a, b, c, e], ![a, b, d, f], ![a, b, c, f], ![a, b, d, e]]
    let y : Fin 6 → ℝ := fun j =>
      br (P j 0 + u j 0) (P j 1 + u j 1) (P j 2 + u j 2) (P j 3 + u j 3)
    |y 0 * y 1 - y 2 * y 3 + y 4 * y 5| ≤ 3 * (((1 + δ) ^ 4 - 1) * (2 + ((1 + δ) ^ 4 - 1))) := by
  intro P y
  set x : Fin 6 → ℝ := fun j => br (P j 0) (P j 1) (P j 2) (P j 3) with hxdef
  have hunit : ∀ j i, ‖P j i‖₂ = 1 := by
    intro j i; fin_cases j <;> fin_cases i <;> simp [P, ha, hb, hc, hd, he, hf]
  have hx : ∀ j, |x j| ≤ 1 := by
    intro j
    refine le_trans (br_hadamard _ _ _ _) (le_of_eq ?_)
    simp [hunit]
  have hε : 0 ≤ (1 + δ) ^ 4 - 1 := by
    have : (1 : ℝ) ≤ (1 + δ) ^ 4 := one_le_pow₀ (by linarith)
    linarith
  have hdiff : ∀ j, |y j - x j| ≤ (1 + δ) ^ 4 - 1 :=
    fun j => br_perturb (P j) (u j) (fun i => (hunit j i).le) (hu j)
  have h0 : x 0 * x 1 - x 2 * x 3 + x 4 * x 5 = 0 := by
    simpa [hxdef, P] using plucker a b c d e f
  exact three_products x y hε hx hdiff h0

/-- **Theorem 7 (a bound for log|J| beyond first order), from the vectors on.** Four brackets
computed from unit representatives `Q i`; the same brackets computed from representatives
within `δ`. If `ε = (1 + δ)^4 - 1` is below every true bracket, the change of
`log|b₁| + log|b₂| - log|b₃| - log|b₄|` (that is, of `log|J|`) is at most
`∑ -log(1 - ε/|b_i|)`, and at most `-4 log(1 - ε/δ_J)` if `ε < δ_J ≤ |b_i|`. -/
theorem logJ_bound_vectors (Q U : Fin 4 → Fin 4 → Fin 4 → ℝ) {δ δJ : ℝ}
    (hQ : ∀ k i, ‖Q k i‖₂ = 1) (hU : ∀ k i, ‖U k i‖₂ ≤ δ)
    (hε : ∀ k, (1 + δ) ^ 4 - 1 < |br (Q k 0) (Q k 1) (Q k 2) (Q k 3)|) :
    let b : Fin 4 → ℝ := fun k => br (Q k 0) (Q k 1) (Q k 2) (Q k 3)
    let b' : Fin 4 → ℝ := fun k =>
      br (Q k 0 + U k 0) (Q k 1 + U k 1) (Q k 2 + U k 2) (Q k 3 + U k 3)
    abs ((Real.log |b' 0| + Real.log |b' 1| - Real.log |b' 2| - Real.log |b' 3|) -
        (Real.log |b 0| + Real.log |b 1| - Real.log |b 2| - Real.log |b 3|))
      ≤ ∑ k, -Real.log (1 - ((1 + δ) ^ 4 - 1) / |b k|) ∧
    ((1 + δ) ^ 4 - 1 < δJ → (∀ k, δJ ≤ |b k|) →
      abs ((Real.log |b' 0| + Real.log |b' 1| - Real.log |b' 2| - Real.log |b' 3|) -
          (Real.log |b 0| + Real.log |b 1| - Real.log |b 2| - Real.log |b 3|))
        ≤ -4 * Real.log (1 - ((1 + δ) ^ 4 - 1) / δJ)) := by
  intro b b'
  exact logJ_bound b b' _ δJ hε
    (fun k => br_perturb (Q k) (U k) (fun i => (hQ k i).le) (hU k))

end Capri
