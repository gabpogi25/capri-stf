/-
Pencil cross-ratios: brackets, gauge invariance, frame identities, and the
Plücker relation as a cross-ratio identity (docs/MATHEMATICS.md, Lemma 2.1,
Theorem 2.5; MATHEMATICS.md Proposition 3.5).
-/
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.LinearCombination

open Matrix

namespace Capri

/-- The bracket `[abcd]`: the determinant of the four vectors as rows. -/
noncomputable def br (a b c d : Fin 4 → ℝ) : ℝ := Matrix.det (Matrix.of ![a, b, c, d])

/-- The pencil cross-ratio `CR_ij(c,d;e,f) = [ijce][ijdf] / ([ijcf][ijde])`. -/
noncomputable def cr (i j c d e f : Fin 4 → ℝ) : ℝ :=
  br i j c e * br i j d f / (br i j c f * br i j d e)

/-- The bracket as an explicit polynomial in the 16 coordinates. -/
theorem br_expand (a b c d : Fin 4 → ℝ) :
    br a b c d =
      a 0 * (b 1 * (c 2 * d 3 - c 3 * d 2) - b 2 * (c 1 * d 3 - c 3 * d 1) + b 3 * (c 1 * d 2 - c 2 * d 1))
    - a 1 * (b 0 * (c 2 * d 3 - c 3 * d 2) - b 2 * (c 0 * d 3 - c 3 * d 0) + b 3 * (c 0 * d 2 - c 2 * d 0))
    + a 2 * (b 0 * (c 1 * d 3 - c 3 * d 1) - b 1 * (c 0 * d 3 - c 3 * d 0) + b 3 * (c 0 * d 1 - c 1 * d 0))
    - a 3 * (b 0 * (c 1 * d 2 - c 2 * d 1) - b 1 * (c 0 * d 2 - c 2 * d 0) + b 2 * (c 0 * d 1 - c 1 * d 0)) := by
  simp [br, Matrix.det_succ_row_zero, Fin.sum_univ_succ, Matrix.det_fin_three, Fin.succAbove]
  ring

/-- A linear map multiplies every bracket by its determinant. -/
theorem br_mulVec (A : Matrix (Fin 4) (Fin 4) ℝ) (a b c d : Fin 4 → ℝ) :
    br (A *ᵥ a) (A *ᵥ b) (A *ᵥ c) (A *ᵥ d) = A.det * br a b c d := by
  have h : Matrix.of ![A *ᵥ a, A *ᵥ b, A *ᵥ c, A *ᵥ d] = Matrix.of ![a, b, c, d] * Aᵀ := by
    ext i j
    fin_cases i <;> simp [Matrix.mul_apply, Matrix.mulVec, dotProduct, mul_comm]
  rw [br, h, Matrix.det_mul, Matrix.det_transpose, br, mul_comm]

/-- Brackets are linear in each argument; in particular they scale. -/
theorem br_smul (s t u v : ℝ) (a b c d : Fin 4 → ℝ) :
    br (s • a) (t • b) (u • c) (v • d) = s * t * u * v * br a b c d := by
  simp only [br_expand, Pi.smul_apply, smul_eq_mul]
  ring

/-- **Lemma 2.1 (gauge invariance).** A pencil cross-ratio, sign included, is unchanged by
an invertible linear map and by nonzero rescalings of the six representatives. -/
theorem cr_gauge (A : Matrix (Fin 4) (Fin 4) ℝ) (hA : A.det ≠ 0)
    (li lj lc ld le lf : ℝ) (hi : li ≠ 0) (hj : lj ≠ 0) (hc : lc ≠ 0) (hd : ld ≠ 0)
    (he : le ≠ 0) (hf : lf ≠ 0) (i j c d e f : Fin 4 → ℝ) :
    cr (li • (A *ᵥ i)) (lj • (A *ᵥ j)) (lc • (A *ᵥ c)) (ld • (A *ᵥ d)) (le • (A *ᵥ e))
        (lf • (A *ᵥ f)) = cr i j c d e f := by
  have k : ∀ (s t u v : ℝ) (p q r w : Fin 4 → ℝ),
      br (s • (A *ᵥ p)) (t • (A *ᵥ q)) (u • (A *ᵥ r)) (v • (A *ᵥ w)) = s * t * u * v * A.det * br p q r w := by
    intro s t u v p q r w
    rw [br_smul, br_mulVec]; ring
  simp only [cr, k]
  have hk : li * lj * lc * ld * le * lf * A.det ^ 2 * (li * lj) ≠ 0 := by
    simp [hi, hj, hc, hd, he, hf, hA]
  rw [show li * lj * lc * le * A.det * br i j c e * (li * lj * ld * lf * A.det * br i j d f)
        = (li * lj * lc * ld * le * lf * A.det ^ 2 * (li * lj)) * (br i j c e * br i j d f) by ring,
      show li * lj * lc * lf * A.det * br i j c f * (li * lj * ld * le * A.det * br i j d e)
        = (li * lj * lc * ld * le * lf * A.det ^ 2 * (li * lj)) * (br i j c f * br i j d e) by ring,
      mul_div_mul_left _ _ hk]

/-- The standard projective frame `e1, e2, e3, e4, (1,1,1,1)`. -/
def e1 : Fin 4 → ℝ := ![1, 0, 0, 0]
def e2 : Fin 4 → ℝ := ![0, 1, 0, 0]
def e3 : Fin 4 → ℝ := ![0, 0, 1, 0]
def e4 : Fin 4 → ℝ := ![0, 0, 0, 1]
def e5 : Fin 4 → ℝ := ![1, 1, 1, 1]

/-- **Proposition 3.5 (frame cross-ratios).** In the standard frame, with `p = (x, y, z, w)`,
`CR_12(3,4;5,p) = z/w`, `CR_13(2,4;5,p) = y/w` and `CR_23(1,4;5,p) = x/w`. (With Lean's
convention `t / 0 = 0` both sides are `0` when `w = 0`, so no hypothesis is needed.) -/
theorem frame_cr (x y z w : ℝ) :
    cr e1 e2 e3 e4 e5 ![x, y, z, w] = z / w ∧
    cr e1 e3 e2 e4 e5 ![x, y, z, w] = y / w ∧
    cr e2 e3 e1 e4 e5 ![x, y, z, w] = x / w := by
  refine ⟨?_, ?_, ?_⟩ <;>
  · simp only [cr, br_expand, e1, e2, e3, e4, e5]
    simp
    try ring_nf
    try field_simp
    try ring_nf

/-- The brackets of a point in the standard frame are its coordinates, up to sign:
`[234p] = -x`, `[134p] = y`, `[124p] = -z`, `[123p] = w`. -/
theorem frame_brackets (x y z w : ℝ) :
    br e2 e3 e4 ![x, y, z, w] = -x ∧ br e1 e3 e4 ![x, y, z, w] = y ∧
    br e1 e2 e4 ![x, y, z, w] = -z ∧ br e1 e2 e3 ![x, y, z, w] = w := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [br_expand, e1, e2, e3, e4]

/-- The Grassmann–Plücker relation of a pencil: `[abcd][abef] - [abce][abdf] + [abcf][abde] = 0`
for any six vectors (Theorem 2.5). -/
theorem plucker (a b c d e f : Fin 4 → ℝ) :
    br a b c d * br a b e f - br a b c e * br a b d f + br a b c f * br a b d e = 0 := by
  simp only [br_expand]
  ring

/-- **Theorem 2.5 (each Plücker relation is a cross-ratio identity).** Whenever the
denominators are nonzero, `CR_ab(c,d;e,f) + CR_ab(c,e;d,f) = 1`. -/
theorem cr_add_cr (a b c d e f : Fin 4 → ℝ) (h1 : br a b c f ≠ 0) (h2 : br a b d e ≠ 0) :
    cr a b c d e f + cr a b c e d f = 1 := by
  have hp := plucker a b c d e f
  have hs : br a b e d = - br a b d e := by simp only [br_expand]; ring
  unfold cr
  rw [hs]
  field_simp
  first
    | linear_combination hp
    | linear_combination -hp
    | linear_combination (br a b c f * br a b d e) * hp
    | linear_combination -(br a b c f * br a b d e) * hp

end Capri
