/-
Linear-algebra cores of the rank theorem (MATHEMATICS.md Lemma 3.6 and Theorem 3.7).
Here: the stabiliser lemma, and the rank-3 block of the lower bound. The Jacobian and orbit
argument that combines them, and the theorem itself, are in `RankFull.lean`.
-/
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith

open Matrix

namespace Capri

/-- **Lemma 3.6 (stabiliser), linear-algebra core.** Let `p 0, …, p 3` be linearly
independent, `q = ∑ c i • p i` with every `c i ≠ 0`, and let `X` satisfy
`X p_i = -μ_i p_i` and `X q = -ν q`. Then every `μ_i = ν` and `X = -ν • 1`. -/
theorem stabiliser (p : Fin 4 → Fin 4 → ℝ) (hp : LinearIndependent ℝ p)
    (c : Fin 4 → ℝ) (hc : ∀ i, c i ≠ 0) (X : Matrix (Fin 4) (Fin 4) ℝ) (μ : Fin 4 → ℝ) (ν : ℝ)
    (hX : ∀ i, X *ᵥ p i = -(μ i) • p i)
    (hq : X *ᵥ (∑ i, c i • p i) = -ν • (∑ i, c i • p i)) :
    (∀ i, μ i = ν) ∧ X = -ν • (1 : Matrix (Fin 4) (Fin 4) ℝ) := by
  have hsum : ∑ i, (c i * (ν - μ i)) • p i = 0 := by
    have h1 : X *ᵥ (∑ i, c i • p i) = ∑ i, (-(c i * μ i)) • p i := by
      rw [Matrix.mulVec_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [Matrix.mulVec_smul, hX i, smul_smul]; ring_nf
    have h2 : -ν • (∑ i, c i • p i) = ∑ i, (-(ν * c i)) • p i := by
      rw [Finset.smul_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [smul_smul]; ring_nf
    have h3 : ∑ i, (-(c i * μ i)) • p i - ∑ i, (-(ν * c i)) • p i = 0 := by
      rw [← h1, ← h2, hq, sub_self]
    rw [← Finset.sum_sub_distrib] at h3
    rw [← h3]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← sub_smul]; congr 1; ring
  have hcoef := (Fintype.linearIndependent_iff.mp hp) _ hsum
  have hμ : ∀ i, μ i = ν := by
    intro i
    have := hcoef i
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h (hc i)
    · linarith
  refine ⟨hμ, ?_⟩
  -- X and -ν • 1 agree on the basis p
  have hspan : Submodule.span ℝ (Set.range p) = ⊤ :=
    hp.span_eq_top_of_card_eq_finrank (by simp)
  have hlin : Matrix.toLin' X = Matrix.toLin' (-ν • (1 : Matrix (Fin 4) (Fin 4) ℝ)) := by
    apply LinearMap.ext_on_range hspan
    intro i
    simp [Matrix.toLin'_apply, hX i, hμ i]
  exact Matrix.toLin'.injective hlin

/-- **Theorem 3.7, lower-bound block.** The gradients of `log|z/w|`, `log|y/w|`, `log|x/w|` with
respect to `(x, y, z, w)` are linearly independent when `x, y, z ≠ 0`. -/
theorem frame_block_rank (x y z w : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    LinearIndependent ℝ ![(![0, 0, 1 / z, -1 / w] : Fin 4 → ℝ), ![0, 1 / y, 0, -1 / w],
      ![1 / x, 0, 0, -1 / w]] := by
  rw [Fintype.linearIndependent_iff]
  intro g hg
  have e0 := congrFun hg 0
  have e1 := congrFun hg 1
  have e2 := congrFun hg 2
  simp [Fin.sum_univ_succ] at e0 e1 e2
  intro i
  fin_cases i <;> simp_all

end Capri
