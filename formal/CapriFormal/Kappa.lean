/-
The conditioning score κ_J and Proposition 5.2 (`κ_J ≤ 8/δ_J ≤ 8/δ_min = 2√(2 L(S))`).

Six points `P : Fin 6 → ℝ⁴` (0-based labels). The invariants are
J1 = [0123][0145]/([0124][0135]), J2 = [0234][1235]/([0235][1234]),
J3 = [0145][2345]/([0245][1345]). The gradient of a function of the 24 coordinates is the vector
of its partial derivatives in `EuclideanSpace ℝ (Fin 6 × Fin 4)`, and
κ_J = max_k ‖∇ log|J_k|‖.

* `br_slot0`..`br_slot3`: a bracket is affine in each slot.
* `partial_logbr`: the partial derivative of log|[abcd]| along coordinate (k, j) is the bracket
  with `e_j` in the slot of `k`, over the bracket (the gradient formula of Proposition 10).
* `cofactor_sq_le`: for unit vectors, `∑_j [e_j b c d]² ≤ 1` (Hadamard).
* `grad_logbr_le`: ‖∇ log|[abcd]|‖ ≤ 2/|[abcd]| at unit representatives.
* `kappa_le`: κ_J ≤ 8/δ_J; `kappa_le_L`: 8/δ_J ≤ 8/δ_min = 2√(2L).
* `no_reverse`: a unit configuration with an unused bracket zero and all 11 used brackets
  nonzero (so L(S) is infinite while κ_J is finite).
-/
import CapriFormal.Hadamard
import CapriFormal.RankFull
import CapriFormal.Strata

open Matrix Finset

namespace Capri

/-- The standard basis vector `e_j` of ℝ⁴. -/
def ev (j : Fin 4) : Fin 4 → ℝ := Pi.single j 1

theorem br_slot0 (x y b c d : Fin 4 → ℝ) (s : ℝ) :
    br (x + s • y) b c d = br x b c d + s * br y b c d := by
  simp only [br_expand, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
theorem br_slot1 (a x y c d : Fin 4 → ℝ) (s : ℝ) :
    br a (x + s • y) c d = br a x c d + s * br a y c d := by
  simp only [br_expand, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
theorem br_slot2 (a b x y d : Fin 4 → ℝ) (s : ℝ) :
    br a b (x + s • y) d = br a b x d + s * br a b y d := by
  simp only [br_expand, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring
theorem br_slot3 (a b c x y : Fin 4 → ℝ) (s : ℝ) :
    br a b c (x + s • y) = br a b c x + s * br a b c y := by
  simp only [br_expand, Pi.add_apply, Pi.smul_apply, smul_eq_mul]; ring

theorem br_swap01 (a b c d : Fin 4 → ℝ) : br a b c d = -br b a c d := by
  simp only [br_expand]; ring
theorem br_cyc2 (a b c d : Fin 4 → ℝ) : br a b c d = br c a b d := by
  simp only [br_expand]; ring
theorem br_cyc3 (a b c d : Fin 4 → ℝ) : br a b c d = -br d a b c := by
  simp only [br_expand]; ring

/-- A bracket is linear in its first slot: `[∑ n_j e_j, b, c, d] = ∑ n_j [e_j b c d]`. -/
theorem br_lin0 (n b c d : Fin 4 → ℝ) : br n b c d = ∑ j, n j * br (ev j) b c d := by
  simp only [br_expand, ev, Fin.sum_univ_four]
  simp [Pi.single_apply]
  ring

/-- **Cofactor bound.** For vectors `b, c, d` of Euclidean norm 1, the cofactor vector
`n_j = [e_j b c d]` has `∑ n_j² ≤ 1` (its norm is vol₃(b, c, d) ≤ 1 by Hadamard). -/
theorem cofactor_sq_le (b c d : Fin 4 → ℝ) (hb : ‖WithLp.toLp 2 b‖ = 1)
    (hc : ‖WithLp.toLp 2 c‖ = 1) (hd : ‖WithLp.toLp 2 d‖ = 1) :
    ∑ j, br (ev j) b c d ^ 2 ≤ 1 := by
  set nv : Fin 4 → ℝ := fun j => br (ev j) b c d
  have hsum : br nv b c d = ∑ j, nv j ^ 2 := by
    rw [br_lin0]; simp [nv, sq]
  have hH := br_hadamard nv b c d
  rw [hb, hc, hd, hsum, mul_one, mul_one, mul_one] at hH
  have hnorm : ‖WithLp.toLp 2 nv‖ ^ 2 = ∑ j, nv j ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
    simp [Real.norm_eq_abs, sq_abs]
  have hS : 0 ≤ ∑ j, nv j ^ 2 := by positivity
  rw [abs_of_nonneg hS] at hH
  nlinarith [norm_nonneg (WithLp.toLp 2 nv)]

/-- The coordinate direction `(k, j)` of configuration space. -/
def ec {n : ℕ} (k : Fin n) (j : Fin 4) : Config n := Pi.single k (ev j)

/-- The gradient over the `4n` coordinates, as a Euclidean vector. -/
noncomputable def grad {n : ℕ} (f : Config n → ℝ) (P : Config n) :
    EuclideanSpace ℝ (Fin n × Fin 4) :=
  WithLp.toLp 2 (fun kj => fderiv ℝ f P (ec kj.1 kj.2))

theorem fderiv_eq_line {n : ℕ} {f : Config n → ℝ} {P : Config n} (hf : DifferentiableAt ℝ f P)
    (v : Config n) {g' : ℝ} (hg : HasDerivAt (fun s : ℝ => f (P + s • v)) g' 0) :
    fderiv ℝ f P v = g' := by
  have hl : HasDerivAt (fun s : ℝ => P + s • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add P
  exact (hf.hasFDerivAt.comp_hasDerivAt_of_eq 0 hl (by simp)).unique hg

/-- The bracket of four points of a configuration. -/
noncomputable def brf {n : ℕ} (a b c d : Fin n) (P : Config n) : ℝ := br (P a) (P b) (P c) (P d)

/-- The bracket with `e_j` in the slot of point `k` (zero if `k` is not in the bracket). -/
noncomputable def brPart {n : ℕ} (a b c d : Fin n) (P : Config n) (k : Fin n) (j : Fin 4) : ℝ :=
  (if k = a then br (ev j) (P b) (P c) (P d) else 0) +
  (if k = b then br (P a) (ev j) (P c) (P d) else 0) +
  (if k = c then br (P a) (P b) (ev j) (P d) else 0) +
  (if k = d then br (P a) (P b) (P c) (ev j) else 0)

theorem line_apply {n : ℕ} (P : Config n) (k : Fin n) (j : Fin 4) (s : ℝ) (m : Fin n) :
    (P + s • ec k j) m = P m + s • (if m = k then ev j else 0) := by
  by_cases h : m = k
  · subst h; simp [ec]
  · simp [ec, h, Pi.single_eq_of_ne h]

/-- A bracket moves affinely along a coordinate direction. -/
theorem brf_line {n : ℕ} {a b c d : Fin n} (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) (P : Config n) (k : Fin n) (j : Fin 4) (s : ℝ) :
    brf a b c d (P + s • ec k j) = brf a b c d P + s * brPart a b c d P k j := by
  simp only [brf, brPart, line_apply]
  by_cases h1 : k = a
  · subst h1
    simp [hab, hac, had, Ne.symm hab, Ne.symm hac, Ne.symm had, br_slot0]
  by_cases h2 : k = b
  · subst h2
    simp [h1, Ne.symm h1, hbc, hbd, Ne.symm hbc, Ne.symm hbd, br_slot1]
  by_cases h3 : k = c
  · subst h3
    simp [h1, h2, Ne.symm h1, Ne.symm h2, hcd, Ne.symm hcd, br_slot2]
  by_cases h4 : k = d
  · subst h4
    simp [h1, h2, h3, Ne.symm h1, Ne.symm h2, Ne.symm h3, br_slot3]
  simp [h1, h2, h3, h4, Ne.symm h1, Ne.symm h2, Ne.symm h3, Ne.symm h4]

theorem brf_differentiable {n : ℕ} (a b c d : Fin n) : Differentiable ℝ (brf a b c d) :=
  br_differentiable a b c d

/-- **The gradient formula** (Jacobi's formula for a bracket): the partial derivative of
`log|[abcd]|` along coordinate `(k, j)` is the bracket with `e_j` in the slot of `k`, over
`[abcd]`. -/
theorem partial_logbr {n : ℕ} {a b c d : Fin n} (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) {P : Config n} (hP : brf a b c d P ≠ 0)
    (k : Fin n) (j : Fin 4) :
    fderiv ℝ (fun Q => Real.log (brf a b c d Q)) P (ec k j)
      = brPart a b c d P k j / brf a b c d P := by
  apply fderiv_eq_line ((brf_differentiable a b c d P).log hP)
  have h := (((hasDerivAt_id (0 : ℝ)).mul_const (brPart a b c d P k j)).const_add
    (brf a b c d P)).log (by simpa using hP)
  simp only [brf_line hab hac had hbc hbd hcd]
  convert h using 1
  all_goals simp

theorem sq_brPart {n : ℕ} {a b c d : Fin n} (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) (P : Config n) (k : Fin n) (j : Fin 4) :
    brPart a b c d P k j ^ 2 =
      (if k = a then br (ev j) (P b) (P c) (P d) ^ 2 else 0) +
      (if k = b then br (P a) (ev j) (P c) (P d) ^ 2 else 0) +
      (if k = c then br (P a) (P b) (ev j) (P d) ^ 2 else 0) +
      (if k = d then br (P a) (P b) (P c) (ev j) ^ 2 else 0) := by
  unfold brPart
  by_cases h1 : k = a
  · subst h1; simp [hab, hac, had]
  by_cases h2 : k = b
  · subst h2; simp [h1, hbc, hbd]
  by_cases h3 : k = c
  · subst h3; simp [h1, h2, hcd]
  by_cases h4 : k = d
  · subst h4; simp [h1, h2, h3]
  simp [h1, h2, h3, h4]

/-- **Gradient of a log-bracket.** At unit representatives,
`‖∇ log|[abcd]|‖ ≤ 2/|[abcd]|` (each of the four cofactor vectors has norm at most 1). -/
theorem grad_logbr_le {n : ℕ} {a b c d : Fin n} (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) {P : Config n}
    (hunit : ∀ m, ‖WithLp.toLp 2 (P m)‖ = 1) (hP : brf a b c d P ≠ 0) :
    ‖grad (fun Q => Real.log (brf a b c d Q)) P‖ ≤ 2 / |brf a b c d P| := by
  set β := brf a b c d P
  have hsq : ∑ kj : Fin n × Fin 4, (brPart a b c d P kj.1 kj.2 / β) ^ 2 ≤ 4 / β ^ 2 := by
    simp_rw [div_pow, ← Finset.sum_div]
    rw [div_le_div_iff_of_pos_right (by positivity)]
    rw [Fintype.sum_prod_type]
    simp_rw [sq_brPart hab hac had hbc hbd hcd]
    rw [Finset.sum_comm]
    simp only [Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    have h0 := cofactor_sq_le (P b) (P c) (P d) (hunit b) (hunit c) (hunit d)
    have h1 := cofactor_sq_le (P a) (P c) (P d) (hunit a) (hunit c) (hunit d)
    have h2 := cofactor_sq_le (P a) (P b) (P d) (hunit a) (hunit b) (hunit d)
    have h3 := cofactor_sq_le (P a) (P b) (P c) (hunit a) (hunit b) (hunit c)
    have e1 : ∀ j, br (P a) (ev j) (P c) (P d) ^ 2 = br (ev j) (P a) (P c) (P d) ^ 2 := by
      intro j; rw [br_swap01]; ring
    have e2 : ∀ j, br (P a) (P b) (ev j) (P d) ^ 2 = br (ev j) (P a) (P b) (P d) ^ 2 := by
      intro j; rw [br_cyc2]
    have e3 : ∀ j, br (P a) (P b) (P c) (ev j) ^ 2 = br (ev j) (P a) (P b) (P c) ^ 2 := by
      intro j; rw [br_cyc3]; ring
    simp only [e1, e2, e3]
    linarith
  have hnorm : ‖grad (fun Q => Real.log (brf a b c d Q)) P‖ =
      Real.sqrt (∑ kj : Fin n × Fin 4, (brPart a b c d P kj.1 kj.2 / β) ^ 2) := by
    rw [grad, EuclideanSpace.norm_eq]
    congr 1
    refine Finset.sum_congr rfl fun kj _ => ?_
    rw [Real.norm_eq_abs, sq_abs]
    change (fderiv ℝ (fun Q => Real.log (brf a b c d Q)) P (ec kj.1 kj.2)) ^ 2 = _
    rw [partial_logbr hab hac had hbc hbd hcd hP]
  rw [hnorm]
  calc Real.sqrt _ ≤ Real.sqrt (4 / β ^ 2) := Real.sqrt_le_sqrt hsq
    _ = 2 / |β| := by
      rw [Real.sqrt_div' _ (by positivity), Real.sqrt_sq_eq_abs,
        show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

/-- The brackets of J1, J2, J3 (0-based): numerator, numerator, denominator, denominator. -/
def Jq : Fin 3 → Fin 4 → Fin 4 → Fin 6 :=
  ![![![0, 1, 2, 3], ![0, 1, 4, 5], ![0, 1, 2, 4], ![0, 1, 3, 5]],
    ![![0, 2, 3, 4], ![1, 2, 3, 5], ![0, 2, 3, 5], ![1, 2, 3, 4]],
    ![![0, 1, 4, 5], ![2, 3, 4, 5], ![0, 2, 4, 5], ![1, 3, 4, 5]]]

theorem Jq_distinct (k : Fin 3) (i : Fin 4) :
    Jq k i 0 ≠ Jq k i 1 ∧ Jq k i 0 ≠ Jq k i 2 ∧ Jq k i 0 ≠ Jq k i 3 ∧
    Jq k i 1 ≠ Jq k i 2 ∧ Jq k i 1 ≠ Jq k i 3 ∧ Jq k i 2 ≠ Jq k i 3 := by
  revert k i; decide

/-- Bracket `i` of `J_k`. -/
noncomputable def bJ (k : Fin 3) (i : Fin 4) (P : Config 6) : ℝ :=
  brf (Jq k i 0) (Jq k i 1) (Jq k i 2) (Jq k i 3) P

/-- `log|J_k|`. -/
noncomputable def logJ (k : Fin 3) (P : Config 6) : ℝ :=
  Real.log (bJ k 0 P * bJ k 1 P / (bJ k 2 P * bJ k 3 P))

/-- κ_J = max_k ‖∇ log|J_k|‖ over the 24 coordinates. -/
noncomputable def kappaJ (P : Config 6) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun k : Fin 3 => ‖grad (logJ k) P‖

/-- δ_J: the smallest |bracket| among those used by J1-J3. -/
noncomputable def deltaJ (P : Config 6) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty fun ki : Fin 3 × Fin 4 => |bJ ki.1 ki.2 P|

/-- δ_min: the smallest |bracket| over all quadruples of distinct points. -/
noncomputable def deltaMin (P : Config 6) : ℝ :=
  (Finset.univ.filter fun q : Fin 4 → Fin 6 => Function.Injective q).inf'
    ⟨![0, 1, 2, 3], by simp; decide⟩ fun q => |brf (q 0) (q 1) (q 2) (q 3) P|

theorem hasDerivAt_logratio {b0 b1 b2 b3 x0 x1 x2 x3 : ℝ} (h0 : b0 ≠ 0) (h1 : b1 ≠ 0)
    (h2 : b2 ≠ 0) (h3 : b3 ≠ 0) :
    HasDerivAt (fun s : ℝ => Real.log ((b0 + s * x0) * (b1 + s * x1) / ((b2 + s * x2) * (b3 + s * x3))))
      (x0 / b0 + x1 / b1 - x2 / b2 - x3 / b3) 0 := by
  have l : ∀ b x : ℝ, HasDerivAt (fun s : ℝ => b + s * x) x 0 := fun b x => by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const x).const_add b
  have h := (((l b0 x0).mul (l b1 x1)).div ((l b2 x2).mul (l b3 x3))
    (by simp [h2, h3])).log (by simp [h0, h1, h2, h3])
  convert h using 1
  all_goals simp only [Pi.div_apply, Pi.mul_apply, zero_mul, add_zero]
  field_simp
  ring

theorem bJ_line (k : Fin 3) (i : Fin 4) (P : Config 6) (m : Fin 6) (j : Fin 4) (s : ℝ) :
    bJ k i (P + s • ec m j) = bJ k i P + s * brPart (Jq k i 0) (Jq k i 1) (Jq k i 2) (Jq k i 3) P m j := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := Jq_distinct k i
  exact brf_line h1 h2 h3 h4 h5 h6 P m j s

theorem logJ_differentiableAt (k : Fin 3) {P : Config 6} (hP : ∀ i, bJ k i P ≠ 0) :
    DifferentiableAt ℝ (logJ k) P := by
  have d : ∀ i, DifferentiableAt ℝ (bJ k i) P := fun i => brf_differentiable _ _ _ _ P
  have h1 : DifferentiableAt ℝ (fun Q => bJ k 0 Q * bJ k 1 Q) P := (d 0).mul (d 1)
  have h2 : DifferentiableAt ℝ (fun Q => bJ k 2 Q * bJ k 3 Q) P := (d 2).mul (d 3)
  have hden : bJ k 2 P * bJ k 3 P ≠ 0 := mul_ne_zero (hP 2) (hP 3)
  have h3 : DifferentiableAt ℝ (fun Q => bJ k 0 Q * bJ k 1 Q / (bJ k 2 Q * bJ k 3 Q)) P := by
    fun_prop (disch := exact hden)
  exact h3.log (div_ne_zero (mul_ne_zero (hP 0) (hP 1)) (mul_ne_zero (hP 2) (hP 3)))

/-- `∇ log|J_k|` is the signed sum of the gradients of its four log-brackets. -/
theorem grad_logJ (k : Fin 3) {P : Config 6} (hP : ∀ i, bJ k i P ≠ 0) :
    grad (logJ k) P = grad (fun Q => Real.log (bJ k 0 Q)) P + grad (fun Q => Real.log (bJ k 1 Q)) P
      - grad (fun Q => Real.log (bJ k 2 Q)) P - grad (fun Q => Real.log (bJ k 3 Q)) P := by
  have hpart : ∀ i (m : Fin 6) (j : Fin 4), fderiv ℝ (fun Q => Real.log (bJ k i Q)) P (ec m j)
      = brPart (Jq k i 0) (Jq k i 1) (Jq k i 2) (Jq k i 3) P m j / bJ k i P := by
    intro i m j
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := Jq_distinct k i
    exact partial_logbr h1 h2 h3 h4 h5 h6 (hP i) m j
  have hJ : ∀ (m : Fin 6) (j : Fin 4), fderiv ℝ (logJ k) P (ec m j) =
      brPart (Jq k 0 0) (Jq k 0 1) (Jq k 0 2) (Jq k 0 3) P m j / bJ k 0 P
      + brPart (Jq k 1 0) (Jq k 1 1) (Jq k 1 2) (Jq k 1 3) P m j / bJ k 1 P
      - brPart (Jq k 2 0) (Jq k 2 1) (Jq k 2 2) (Jq k 2 3) P m j / bJ k 2 P
      - brPart (Jq k 3 0) (Jq k 3 1) (Jq k 3 2) (Jq k 3 3) P m j / bJ k 3 P := by
    intro m j
    apply fderiv_eq_line (logJ_differentiableAt k hP)
    simp only [logJ, bJ_line]
    exact hasDerivAt_logratio (hP 0) (hP 1) (hP 2) (hP 3)
  ext ⟨m, j⟩
  simp only [grad, PiLp.add_apply, PiLp.sub_apply, PiLp.toLp_apply, hJ, hpart]

/-- **Proposition 5.2, first inequality.** At unit representatives with every bracket of
J1-J3 nonzero, κ_J ≤ 8/δ_J. -/
theorem kappa_le {P : Config 6} (hunit : ∀ m, ‖WithLp.toLp 2 (P m)‖ = 1) (hδ : 0 < deltaJ P) :
    kappaJ P ≤ 8 / deltaJ P := by
  have hle : ∀ k i, deltaJ P ≤ |bJ k i P| := fun k i =>
    Finset.inf'_le _ (Finset.mem_univ (k, i))
  have hne : ∀ k i, bJ k i P ≠ 0 := fun k i h => by
    have := hle k i; rw [h, abs_zero] at this; linarith
  have hb : ∀ k i, ‖grad (fun Q => Real.log (bJ k i Q)) P‖ ≤ 2 / deltaJ P := by
    intro k i
    obtain ⟨h1, h2, h3, h4, h5, h6⟩ := Jq_distinct k i
    refine le_trans (grad_logbr_le h1 h2 h3 h4 h5 h6 hunit (hne k i)) ?_
    exact div_le_div_of_nonneg_left (by norm_num) hδ (hle k i)
  refine Finset.sup'_le _ _ fun k _ => ?_
  rw [grad_logJ k (hne k)]
  calc _ ≤ ‖grad (fun Q => Real.log (bJ k 0 Q)) P‖ + ‖grad (fun Q => Real.log (bJ k 1 Q)) P‖
        + ‖grad (fun Q => Real.log (bJ k 2 Q)) P‖ + ‖grad (fun Q => Real.log (bJ k 3 Q)) P‖ := by
        have e1 := norm_sub_le (grad (fun Q => Real.log (bJ k 0 Q)) P
          + grad (fun Q => Real.log (bJ k 1 Q)) P - grad (fun Q => Real.log (bJ k 2 Q)) P)
          (grad (fun Q => Real.log (bJ k 3 Q)) P)
        have e2 := norm_sub_le (grad (fun Q => Real.log (bJ k 0 Q)) P
          + grad (fun Q => Real.log (bJ k 1 Q)) P) (grad (fun Q => Real.log (bJ k 2 Q)) P)
        have e3 := norm_add_le (grad (fun Q => Real.log (bJ k 0 Q)) P)
          (grad (fun Q => Real.log (bJ k 1 Q)) P)
        linarith
    _ ≤ 2 / deltaJ P + 2 / deltaJ P + 2 / deltaJ P + 2 / deltaJ P := by
        gcongr <;> exact hb k _
    _ = 8 / deltaJ P := by ring

/-- **Proposition 5.2, the chain.** `8/δ_J ≤ 8/δ_min = 2√(2 L(S))` with `L(S) = 8/δ_min²`. -/
theorem kappa_le_L {P : Config 6} (hmin : 0 < deltaMin P) :
    8 / deltaJ P ≤ 8 / deltaMin P ∧ 8 / deltaMin P = 2 * Real.sqrt (2 * (8 / deltaMin P ^ 2)) := by
  have hle : deltaMin P ≤ deltaJ P := by
    refine Finset.le_inf' _ _ fun ki _ => ?_
    have hinj : Function.Injective (Jq ki.1 ki.2) := by
      revert ki; decide
    exact Finset.inf'_le _ (by simpa using hinj)
  constructor
  · exact div_le_div_of_nonneg_left (by norm_num) hmin hle
  · have h : 2 * (8 / deltaMin P ^ 2) = (4 / deltaMin P) ^ 2 := by field_simp; ring
    rw [h, Real.sqrt_sq (by positivity)]
    ring

/-- The example configuration: the standard frame, `(1/2)(1,1,1,1)` and `(1/3, 2/3, 2/3, 0)`. -/
noncomputable def P0 : Config 6 :=
  ![![1, 0, 0, 0], ![0, 1, 0, 0], ![0, 0, 1, 0], ![0, 0, 0, 1], ![1/2, 1/2, 1/2, 1/2],
    ![1/3, 2/3, 2/3, 0]]

/-- **Proposition 5.2, no reverse bound.** `P0` has unit points, all 11 brackets of J1-J3
nonzero (so δ_J > 0 and κ_J ≤ 8/δ_J is finite), and the unused bracket [1236] equal to zero
(so δ_min = 0 and L(S) is infinite). -/
theorem no_reverse :
    (∀ m, ‖WithLp.toLp 2 (P0 m)‖ = 1) ∧ (∀ k i, bJ k i P0 ≠ 0) ∧ brf 0 1 2 5 P0 = 0 := by
  refine ⟨?_, ?_, ?_⟩
  · intro m
    rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one]
    fin_cases m <;> simp [P0, Fin.sum_univ_four] <;> norm_num
  · intro k i
    fin_cases k <;> fin_cases i <;> simp [bJ, Jq, brf, br_expand, P0] <;> norm_num
  · simp [brf, br_expand, P0]

end Capri
