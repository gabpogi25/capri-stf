/-
Theorem 4 (MATHEMATICS.md Theorem 3.7): the Jacobian of the pencil map has rank exactly 3n - 15 on U_n.

A configuration is `P : Fin n → (Fin 4 → ℝ)`. A pencil is an ordered 6-tuple of distinct point
indices `t`, with cross-ratio `CR_{t0 t1}(t2, t3; t4, t5)`. The pencil map over a family
`pen : ι → (Fin 6 → Fin n)` is `Φ(P) = (log|CR_{pen i}(P)|)_i` (Lean's `Real.log` satisfies
`log x = log |x|`).

* `Phi_act`: Φ is constant on gauge orbits (Lemma 2.1 for every pencil).
* `Phi_differentiableAt`: Φ is differentiable on U_n.
* `fderiv_orbit`: the derivative of Φ vanishes on the tangent space of the orbit,
  `(X, μ) ↦ (X p_k + μ_k p_k)_k`.
* `orbitMap_ker`: that map has kernel spanned by `(I, -1)` (Lemma 3.6, Cramer's rule included),
  so its image has dimension `n + 15`.
* `rank_le`: rank DΦ(P) ≤ 3n - 15.
* `frame_normalise`: with `p_4 = ∑ c_i p_i`, the matrix sending `c_i p_i` to `e_i` turns the frame
  pencils through `p_0, …, p_4` into `z/w`, `y/w`, `x/w` (Proposition 3.5).
* `rank_ge`: rank DΦ(P) ≥ 3(n - 5) whenever the family contains the 3(n - 5) frame pencils.
* `rank_pencil_map`: rank DΦ(P) = 3n - 15; `rank_all_pencils`: the same for all pencils.
-/
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Topology.Instances.Matrix
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import CapriFormal.Pencil
import CapriFormal.Rank

open Matrix Filter Topology Module

namespace Capri

variable {n : ℕ}

/-- Configurations of `n` points of ℝ⁴. -/
abbrev Config (n : ℕ) := Fin n → Fin 4 → ℝ

/-- `U_n`: every bracket of four distinct points is nonzero. -/
def Un (n : ℕ) : Set (Config n) :=
  {P | ∀ a b c d : Fin n, a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    br (P a) (P b) (P c) (P d) ≠ 0}

/-- The cross-ratio `CR_{t0 t1}(t2, t3; t4, t5)` of the pencil `t`. -/
noncomputable def crAt (t : Fin 6 → Fin n) (P : Config n) : ℝ :=
  cr (P (t 0)) (P (t 1)) (P (t 2)) (P (t 3)) (P (t 4)) (P (t 5))

/-- The pencil map `Φ(P) = (log|CR_{pen i}(P)|)_i`. -/
noncomputable def Phi {ι : Type*} (pen : ι → Fin 6 → Fin n) (P : Config n) : ι → ℝ :=
  fun i => Real.log (crAt (pen i) P)

/-- The gauge action: `p_k ↦ l_k A p_k`. -/
def act (A : Matrix (Fin 4) (Fin 4) ℝ) (l : Fin n → ℝ) (Q : Config n) : Config n :=
  fun k => l k • (A *ᵥ Q k)

/-- **Lemma 2.1 for the pencil map.** Φ is constant on gauge orbits. -/
theorem Phi_act {ι : Type*} (pen : ι → Fin 6 → Fin n) (A : Matrix (Fin 4) (Fin 4) ℝ)
    (hA : A.det ≠ 0) (l : Fin n → ℝ) (hl : ∀ k, l k ≠ 0) (Q : Config n) :
    Phi pen (act A l Q) = Phi pen Q := by
  funext i
  simp only [Phi, crAt, act]
  rw [cr_gauge A hA _ _ _ _ _ _ (hl _) (hl _) (hl _) (hl _) (hl _) (hl _)]

/-- A bracket of four points is a polynomial in the coordinates, hence differentiable. -/
theorem br_differentiable (a b c d : Fin n) :
    Differentiable ℝ (fun Q : Config n => br (Q a) (Q b) (Q c) (Q d)) := by
  simp only [br_expand]
  fun_prop

theorem distinct_of_injective {t : Fin 6 → Fin n} (ht : Function.Injective t) {i j : Fin 6}
    (h : i ≠ j) : t i ≠ t j := fun e => h (ht e)

/-- On U_n the cross-ratio of a pencil of distinct points is nonzero and differentiable. -/
theorem crAt_ne_zero {t : Fin 6 → Fin n} (ht : Function.Injective t) {P : Config n}
    (hP : P ∈ Un n) : crAt t P ≠ 0 := by
  have d := fun i j (h : i ≠ j) => distinct_of_injective ht h
  simp only [crAt, cr]
  refine div_ne_zero (mul_ne_zero ?_ ?_) (mul_ne_zero ?_ ?_) <;>
    exact hP _ _ _ _ (d _ _ (by decide)) (d _ _ (by decide)) (d _ _ (by decide))
      (d _ _ (by decide)) (d _ _ (by decide)) (d _ _ (by decide))

theorem crAt_differentiableAt {t : Fin 6 → Fin n} (ht : Function.Injective t) {P : Config n}
    (hP : P ∈ Un n) : DifferentiableAt ℝ (crAt t) P := by
  have d := fun i j (h : i ≠ j) => distinct_of_injective ht h
  have hden : br (P (t 0)) (P (t 1)) (P (t 2)) (P (t 5)) * br (P (t 0)) (P (t 1)) (P (t 3)) (P (t 4))
      ≠ 0 := by
    refine mul_ne_zero ?_ ?_ <;>
      exact hP _ _ _ _ (d _ _ (by decide)) (d _ _ (by decide)) (d _ _ (by decide))
        (d _ _ (by decide)) (d _ _ (by decide)) (d _ _ (by decide))
  have h : crAt t = fun Q : Config n =>
      br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 4)) * br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 5)) /
        (br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 5)) * br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 4))) :=
    rfl
  rw [h]
  simp only [br_expand] at hden ⊢
  fun_prop (disch := exact hden)

/-- Φ is differentiable on U_n. -/
theorem Phi_differentiableAt {ι : Type*} [Fintype ι] (pen : ι → Fin 6 → Fin n)
    (hpen : ∀ i, Function.Injective (pen i)) {P : Config n} (hP : P ∈ Un n) :
    DifferentiableAt ℝ (Phi pen) P := by
  rw [differentiableAt_pi]
  intro i
  exact (crAt_differentiableAt (hpen i) hP).log (crAt_ne_zero (hpen i) hP)

/-- The differential at the identity of the orbit map `g ↦ g·P`. -/
def orbitMap (P : Config n) : (Matrix (Fin 4) (Fin 4) ℝ × (Fin n → ℝ)) →ₗ[ℝ] Config n where
  toFun xm := fun k => xm.1 *ᵥ P k + xm.2 k • P k
  map_add' a b := by
    funext k
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Matrix.add_mulVec, add_smul]
    abel
  map_smul' c a := by
    funext k
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply,
      Matrix.smul_mulVec, smul_add, mul_smul]

/-- **Upper bound, analytic step.** DΦ(P) vanishes on the tangent space of the gauge orbit. -/
theorem fderiv_orbit {ι : Type*} [Fintype ι] (pen : ι → Fin 6 → Fin n)
    (hpen : ∀ i, Function.Injective (pen i)) {P : Config n} (hP : P ∈ Un n)
    (X : Matrix (Fin 4) (Fin 4) ℝ) (μ : Fin n → ℝ) :
    fderiv ℝ (Phi pen) P (orbitMap P (X, μ)) = 0 := by
  set v := orbitMap P (X, μ)
  set w : Config n := fun k => μ k • (X *ᵥ P k)
  set γ : ℝ → Config n := fun s => P + s • v + s ^ 2 • w
  have hγ : ∀ s, γ s = act (1 + s • X) (fun k => 1 + s * μ k) P := by
    intro s
    funext k j
    simp only [γ, v, w, act, orbitMap, LinearMap.coe_mk, AddHom.coe_mk, Pi.add_apply,
      Pi.smul_apply, smul_eq_mul, Matrix.add_mulVec, Matrix.one_mulVec, Matrix.smul_mulVec]
    ring
  have hdet : ∀ᶠ s in 𝓝 (0 : ℝ), (1 + s • X).det ≠ 0 := by
    have hc : Continuous fun s : ℝ => (1 + s • X).det := by fun_prop
    exact hc.continuousAt.eventually_ne (by simp)
  have hl : ∀ᶠ s in 𝓝 (0 : ℝ), ∀ k, 1 + s * μ k ≠ 0 := by
    refine Filter.eventually_all.2 fun k => ?_
    have hc : Continuous fun s : ℝ => 1 + s * μ k := by fun_prop
    exact hc.continuousAt.eventually_ne (by simp)
  have heq : (fun s => Phi pen (γ s)) =ᶠ[𝓝 0] fun _ => Phi pen P := by
    filter_upwards [hdet, hl] with s hs hs'
    rw [hγ s, Phi_act pen _ hs _ hs' P]
  have hdγ : HasDerivAt γ v 0 := by
    have h1 := ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add P
    have h2 := (hasDerivAt_pow 2 (0 : ℝ)).smul_const w
    convert h1.add h2 using 1
    · funext s; simp [γ]
    · simp
  have hD := (Phi_differentiableAt pen hpen hP).hasFDerivAt.comp_hasDerivAt_of_eq 0 hdγ
    (by simp [γ])
  have hc : HasDerivAt (fun s => Phi pen (γ s)) 0 0 :=
    (hasDerivAt_const (0 : ℝ) (Phi pen P)).congr_of_eventuallyEq heq
  exact hD.unique hc

/-- The points `0, …, 4` of a configuration with `n ≥ 5` points. -/
def pt (hn : 5 ≤ n) (m : Fin 5) : Fin n := Fin.castLE hn m

theorem pt_ne (hn : 5 ≤ n) {a b : Fin 5} (h : a ≠ b) : pt hn a ≠ pt hn b :=
  fun e => h (Fin.castLE_injective hn e)

/-- **Cramer's rule** for brackets: with `q = ∑ c_i p_i`, putting `q` in slot `i` of
`[p_0 p_1 p_2 p_3]` multiplies the bracket by `c_i`. -/
theorem br_cramer (p : Fin 4 → Fin 4 → ℝ) (c : Fin 4 → ℝ) :
    let q := ∑ i, c i • p i
    br q (p 1) (p 2) (p 3) = c 0 * br (p 0) (p 1) (p 2) (p 3) ∧
    br (p 0) q (p 2) (p 3) = c 1 * br (p 0) (p 1) (p 2) (p 3) ∧
    br (p 0) (p 1) q (p 3) = c 2 * br (p 0) (p 1) (p 2) (p 3) ∧
    br (p 0) (p 1) (p 2) q = c 3 * br (p 0) (p 1) (p 2) (p 3) := by
  intro q
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
  · simp only [q, br_expand, Finset.sum_apply, Fin.sum_univ_four, Pi.add_apply, Pi.smul_apply,
      smul_eq_mul]
    ring

/-- Points `0..3` of a configuration in U_n are linearly independent. -/
theorem frame_indep (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    LinearIndependent ℝ (fun i : Fin 4 => P (pt hn i.castSucc)) := by
  have hdet := hP (pt hn 0) (pt hn 1) (pt hn 2) (pt hn 3) (pt_ne hn (by decide))
    (pt_ne hn (by decide)) (pt_ne hn (by decide)) (pt_ne hn (by decide)) (pt_ne hn (by decide))
    (pt_ne hn (by decide))
  have h := Matrix.linearIndependent_rows_of_det_ne_zero hdet
  convert h using 1
  funext i
  fin_cases i <;> rfl

/-- **Lemma 3.6, the Cramer step.** In U_n, `p_4 = ∑ c_i p_i` with every `c_i ≠ 0`. -/
theorem frame_coeffs (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    ∃ c : Fin 4 → ℝ, (∀ i, c i ≠ 0) ∧ P (pt hn 4) = ∑ i, c i • P (pt hn i.castSucc) := by
  set p : Fin 4 → Fin 4 → ℝ := fun i => P (pt hn i.castSucc) with hp
  have hli : LinearIndependent ℝ p := frame_indep hn hP
  set b := basisOfLinearIndependentOfCardEqFinrank hli (by simp)
  set c : Fin 4 → ℝ := fun i => b.repr (P (pt hn 4)) i
  have hsum : P (pt hn 4) = ∑ i, c i • p i := by
    conv_lhs => rw [← b.sum_repr (P (pt hn 4))]
    simp [b, c, coe_basisOfLinearIndependentOfCardEqFinrank]
  refine ⟨c, ?_, hsum⟩
  obtain ⟨h0, h1, h2, h3⟩ := br_cramer p c
  rw [← hsum] at h0 h1 h2 h3
  have d := fun a b (h : a ≠ b) => pt_ne hn h
  intro i h
  fin_cases i
  · exact hP _ _ _ _ (d 4 1 (by decide)) (d 4 2 (by decide)) (d 4 3 (by decide))
      (d 1 2 (by decide)) (d 1 3 (by decide)) (d 2 3 (by decide)) (h0.trans (mul_eq_zero_of_left h _))
  · exact hP _ _ _ _ (d 0 4 (by decide)) (d 0 2 (by decide)) (d 0 3 (by decide))
      (d 4 2 (by decide)) (d 4 3 (by decide)) (d 2 3 (by decide)) (h1.trans (mul_eq_zero_of_left h _))
  · exact hP _ _ _ _ (d 0 1 (by decide)) (d 0 4 (by decide)) (d 0 3 (by decide))
      (d 1 4 (by decide)) (d 1 3 (by decide)) (d 4 3 (by decide)) (h2.trans (mul_eq_zero_of_left h _))
  · exact hP _ _ _ _ (d 0 1 (by decide)) (d 0 2 (by decide)) (d 0 4 (by decide))
      (d 1 2 (by decide)) (d 1 4 (by decide)) (d 2 4 (by decide)) (h3.trans (mul_eq_zero_of_left h _))

/-- In U_n (n ≥ 5) no point is zero. -/
theorem Un_ne_zero (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) (k : Fin n) : P k ≠ 0 := by
  intro h0
  have key : ∃ a b c : Fin n, k ≠ a ∧ k ≠ b ∧ k ≠ c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c := by
    rcases (by omega : k.val = 0 ∨ k.val = 1 ∨ k.val = 2 ∨ 3 ≤ k.val) with h | h | h | h
    · refine ⟨⟨1, by omega⟩, ⟨2, by omega⟩, ⟨3, by omega⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
        simp only [ne_eq, Fin.ext_iff] <;> omega
    · refine ⟨⟨0, by omega⟩, ⟨2, by omega⟩, ⟨3, by omega⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
        simp only [ne_eq, Fin.ext_iff] <;> omega
    · refine ⟨⟨0, by omega⟩, ⟨1, by omega⟩, ⟨3, by omega⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
        simp only [ne_eq, Fin.ext_iff] <;> omega
    · refine ⟨⟨0, by omega⟩, ⟨1, by omega⟩, ⟨2, by omega⟩, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
        simp only [ne_eq, Fin.ext_iff] <;> omega
  obtain ⟨a, b, c, h1, h2, h3, h4, h5, h6⟩ := key
  exact hP k a b c h1 h2 h3 h4 h5 h6 (by simp [h0, br_expand])

/-- **Lemma 3.6 (stabiliser).** For P in U_n the kernel of the orbit differential is spanned by
`(I, -1)`. -/
theorem orbitMap_ker (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    LinearMap.ker (orbitMap P) =
      Submodule.span ℝ {((1 : Matrix (Fin 4) (Fin 4) ℝ), fun _ : Fin n => (-1 : ℝ))} := by
  apply le_antisymm
  · rintro ⟨X, μ⟩ hxm
    have hk : ∀ k, X *ᵥ P k = -(μ k) • P k := by
      intro k
      have h := congrFun (LinearMap.mem_ker.1 hxm) k
      simp only [orbitMap, LinearMap.coe_mk, AddHom.coe_mk, Pi.zero_apply] at h
      rw [neg_smul]
      exact eq_neg_of_add_eq_zero_left h
    obtain ⟨c, hc, h4⟩ := frame_coeffs hn hP
    obtain ⟨hμ, hX⟩ := stabiliser _ (frame_indep hn hP) c hc X (fun i => μ (pt hn i.castSucc))
      (μ (pt hn 4)) (fun i => hk _) (by rw [← h4]; exact hk _)
    have hall : ∀ k, μ k = μ (pt hn 4) := by
      intro k
      have h1 := hk k
      rw [hX, Matrix.smul_mulVec, Matrix.one_mulVec] at h1
      have h2 : (μ k - μ (pt hn 4)) • P k = 0 := by
        rw [sub_smul]
        have := congrArg (fun v => -v) h1
        simp only [neg_smul, neg_neg] at this
        rw [this, sub_self]
      rcases smul_eq_zero.1 h2 with h | h
      · linarith
      · exact absurd h (Un_ne_zero hn hP k)
    rw [Submodule.mem_span_singleton]
    refine ⟨-μ (pt hn 4), ?_⟩
    ext1
    · simp [hX]
    · funext k; simp [hall k]
  · rw [Submodule.span_le, Set.singleton_subset_iff]
    show _ ∈ LinearMap.ker (orbitMap P)
    rw [LinearMap.mem_ker]
    funext k
    simp [orbitMap]

/-- The orbit through a point of U_n has dimension `n + 15`. -/
theorem orbitMap_range_finrank (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    finrank ℝ (LinearMap.range (orbitMap P)) = n + 15 := by
  have h := LinearMap.finrank_range_add_finrank_ker (orbitMap P)
  have hne : ((1 : Matrix (Fin 4) (Fin 4) ℝ), fun _ : Fin n => (-1 : ℝ)) ≠ 0 :=
    fun e => one_ne_zero (congrArg Prod.fst e)
  rw [orbitMap_ker hn hP, finrank_span_singleton hne] at h
  simp [Module.finrank_prod, Module.finrank_matrix] at h
  omega

/-- **Theorem 3.7, upper bound.** rank DΦ(P) ≤ 3n - 15. -/
theorem rank_le (hn : 5 ≤ n) {ι : Type*} [Fintype ι] (pen : ι → Fin 6 → Fin n)
    (hpen : ∀ i, Function.Injective (pen i)) {P : Config n} (hP : P ∈ Un n) :
    finrank ℝ (LinearMap.range (fderiv ℝ (Phi pen) P : Config n →ₗ[ℝ] ι → ℝ)) ≤ 3 * n - 15 := by
  have hsub : LinearMap.range (orbitMap P) ≤
      LinearMap.ker (fderiv ℝ (Phi pen) P : Config n →ₗ[ℝ] ι → ℝ) := by
    rintro v ⟨xm, rfl⟩
    exact fderiv_orbit pen hpen hP xm.1 xm.2
  have h1 := Submodule.finrank_mono hsub
  have h2 := LinearMap.finrank_range_add_finrank_ker (fderiv ℝ (Phi pen) P : Config n →ₗ[ℝ] ι → ℝ)
  rw [orbitMap_range_finrank hn hP] at h1
  have h3 : finrank ℝ (Config n) = n * 4 := by
    simp [Module.finrank_pi_fintype]
  omega

/-- Pencil cross-ratios are unchanged by an invertible linear map. -/
theorem cr_lin (A : Matrix (Fin 4) (Fin 4) ℝ) (hA : A.det ≠ 0) (i j c d e f : Fin 4 → ℝ) :
    cr (A *ᵥ i) (A *ᵥ j) (A *ᵥ c) (A *ᵥ d) (A *ᵥ e) (A *ᵥ f) = cr i j c d e f := by
  simpa using cr_gauge A hA 1 1 1 1 1 1 one_ne_zero one_ne_zero one_ne_zero one_ne_zero
    one_ne_zero one_ne_zero i j c d e f

/-- Pencil cross-ratios are unchanged by nonzero rescalings of the six points. -/
theorem cr_scale {li lj lc ld le lf : ℝ} (hi : li ≠ 0) (hj : lj ≠ 0) (hc : lc ≠ 0) (hd : ld ≠ 0)
    (he : le ≠ 0) (hf : lf ≠ 0) (i j c d e f : Fin 4 → ℝ) :
    cr (li • i) (lj • j) (lc • c) (ld • d) (le • e) (lf • f) = cr i j c d e f := by
  simpa using cr_gauge 1 (by simp) li lj lc ld le lf hi hj hc hd he hf i j c d e f

theorem vec_eta (v : Fin 4 → ℝ) : v = ![v 0, v 1, v 2, v 3] := by
  ext m; fin_cases m <;> rfl

/-- **Frame normalisation.** For P in U_n there is an invertible `A` such that the three frame
pencils through `p_0, …, p_4` and a point `q` are `z/w`, `y/w`, `x/w` with `(x, y, z, w) = A q`
(Proposition 3.5 transported by the gauge), and each coordinate of `A q` is nonzero when the
matching bracket of `q` is. -/
theorem frame_normalise (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    ∃ A : Matrix (Fin 4) (Fin 4) ℝ, A.det ≠ 0 ∧
    (∃ s : Fin 4 → ℝ, (∀ i, s i ≠ 0) ∧ ∀ i, A *ᵥ P (pt hn i.castSucc) = s i • Pi.single i 1) ∧
    A *ᵥ P (pt hn 4) = e5 ∧
    (∀ q, cr (P (pt hn 0)) (P (pt hn 1)) (P (pt hn 2)) (P (pt hn 3)) (P (pt hn 4)) q
      = (A *ᵥ q) 2 / (A *ᵥ q) 3) ∧
    (∀ q, cr (P (pt hn 0)) (P (pt hn 2)) (P (pt hn 1)) (P (pt hn 3)) (P (pt hn 4)) q
      = (A *ᵥ q) 1 / (A *ᵥ q) 3) ∧
    (∀ q, cr (P (pt hn 1)) (P (pt hn 2)) (P (pt hn 0)) (P (pt hn 3)) (P (pt hn 4)) q
      = (A *ᵥ q) 0 / (A *ᵥ q) 3) ∧
    (∀ q, br (P (pt hn 1)) (P (pt hn 2)) (P (pt hn 3)) q ≠ 0 → (A *ᵥ q) 0 ≠ 0) ∧
    (∀ q, br (P (pt hn 0)) (P (pt hn 2)) (P (pt hn 3)) q ≠ 0 → (A *ᵥ q) 1 ≠ 0) ∧
    (∀ q, br (P (pt hn 0)) (P (pt hn 1)) (P (pt hn 3)) q ≠ 0 → (A *ᵥ q) 2 ≠ 0) ∧
    (∀ q, br (P (pt hn 0)) (P (pt hn 1)) (P (pt hn 2)) q ≠ 0 → (A *ᵥ q) 3 ≠ 0) := by
  obtain ⟨c, hc, h4⟩ := frame_coeffs hn hP
  set p : Fin 4 → Fin 4 → ℝ := fun i => P (pt hn i.castSucc) with hp
  have hp0 : P (pt hn 0) = p 0 := rfl
  have hp1 : P (pt hn 1) = p 1 := rfl
  have hp2 : P (pt hn 2) = p 2 := rfl
  have hp3 : P (pt hn 3) = p 3 := rfl
  have h0123 : br (p 0) (p 1) (p 2) (p 3) ≠ 0 :=
    hP _ _ _ _ (pt_ne hn (by decide)) (pt_ne hn (by decide)) (pt_ne hn (by decide))
      (pt_ne hn (by decide)) (pt_ne hn (by decide)) (pt_ne hn (by decide))
  set B : Matrix (Fin 4) (Fin 4) ℝ := Matrix.of fun m i => c i * p i m with hBdef
  have hBt : Bᵀ = Matrix.of ![c 0 • p 0, c 1 • p 1, c 2 • p 2, c 3 • p 3] := by
    ext i m; fin_cases i <;> simp [B]
  have hB : B.det ≠ 0 := by
    rw [← Matrix.det_transpose, hBt]
    change br (c 0 • p 0) (c 1 • p 1) (c 2 • p 2) (c 3 • p 3) ≠ 0
    rw [br_smul]
    exact mul_ne_zero (mul_ne_zero (mul_ne_zero (mul_ne_zero (hc 0) (hc 1)) (hc 2)) (hc 3)) h0123
  set A := B⁻¹ with hAdef
  have hAB : A * B = 1 := Matrix.nonsing_inv_mul B (isUnit_iff_ne_zero.2 hB)
  have hA : A.det ≠ 0 := by
    intro h
    have := congrArg Matrix.det hAB
    rw [Matrix.det_mul, h, zero_mul, Matrix.det_one] at this
    exact zero_ne_one this
  have hBe : ∀ i, B *ᵥ Pi.single i 1 = c i • p i := by
    intro i; ext m
    simp [B, Matrix.mulVec, dotProduct, Pi.single_apply]
  have hAp : ∀ i, A *ᵥ p i = (c i)⁻¹ • Pi.single i 1 := by
    intro i
    have h : A *ᵥ (c i • p i) = Pi.single i 1 := by
      rw [← hBe, Matrix.mulVec_mulVec, hAB, Matrix.one_mulVec]
    rw [Matrix.mulVec_smul] at h
    rw [← h, smul_smul, inv_mul_cancel₀ (hc i), one_smul]
  have hA4 : A *ᵥ P (pt hn 4) = e5 := by
    rw [h4, Matrix.mulVec_sum]
    have h : ∀ i, A *ᵥ (c i • P (pt hn i.castSucc)) = Pi.single i 1 := by
      intro i
      change A *ᵥ (c i • p i) = _
      rw [Matrix.mulVec_smul, hAp, smul_smul, mul_inv_cancel₀ (hc i), one_smul]
    simp only [h]
    ext m
    fin_cases m <;> simp [e5, Fin.sum_univ_four, Pi.single_apply]
  have hA0 : A *ᵥ p 0 = (c 0)⁻¹ • e1 := by
    rw [hAp]; congr 1; ext m; fin_cases m <;> simp [e1]
  have hA1 : A *ᵥ p 1 = (c 1)⁻¹ • e2 := by
    rw [hAp]; congr 1; ext m; fin_cases m <;> simp [e2]
  have hA2 : A *ᵥ p 2 = (c 2)⁻¹ • e3 := by
    rw [hAp]; congr 1; ext m; fin_cases m <;> simp [e3]
  have hA3 : A *ᵥ p 3 = (c 3)⁻¹ • e4 := by
    rw [hAp]; congr 1; ext m; fin_cases m <;> simp [e4]
  have hi := fun i => inv_ne_zero (hc i)
  have tr : ∀ (a b d : Fin 4 → ℝ) (q : Fin 4 → ℝ), cr a b d (p 3) (P (pt hn 4)) q
      = cr (A *ᵥ a) (A *ᵥ b) (A *ᵥ d) ((c 3)⁻¹ • e4) e5 (A *ᵥ q) := by
    intro a b d q
    rw [← hA3, ← hA4, cr_lin A hA]
  have tbr : ∀ (a b d q : Fin 4 → ℝ),
      br (A *ᵥ a) (A *ᵥ b) (A *ᵥ d) (A *ᵥ q) = A.det * br a b d q := br_mulVec A
  refine ⟨A, hA, ⟨fun i => (c i)⁻¹, hi, hAp⟩, hA4, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro q
    rw [hp0, hp1, hp2, hp3, tr, hA0, hA1, hA2]
    have hs := cr_scale (hi 0) (hi 1) (hi 2) (hi 3) one_ne_zero one_ne_zero e1 e2 e3 e4 e5 (A *ᵥ q)
    rw [one_smul, one_smul] at hs
    rw [hs]
    conv_lhs => rw [vec_eta (A *ᵥ q)]
    exact (frame_cr _ _ _ _).1
  · intro q
    rw [hp0, hp1, hp2, hp3, tr, hA0, hA1, hA2]
    have hs := cr_scale (hi 0) (hi 2) (hi 1) (hi 3) one_ne_zero one_ne_zero e1 e3 e2 e4 e5 (A *ᵥ q)
    rw [one_smul, one_smul] at hs
    rw [hs]
    conv_lhs => rw [vec_eta (A *ᵥ q)]
    exact (frame_cr _ _ _ _).2.1
  · intro q
    rw [hp0, hp1, hp2, hp3, tr, hA0, hA1, hA2]
    have hs := cr_scale (hi 1) (hi 2) (hi 0) (hi 3) one_ne_zero one_ne_zero e2 e3 e1 e4 e5 (A *ᵥ q)
    rw [one_smul, one_smul] at hs
    rw [hs]
    conv_lhs => rw [vec_eta (A *ᵥ q)]
    exact (frame_cr _ _ _ _).2.2
  · intro q hq h0
    have h := tbr (p 1) (p 2) (p 3) q
    rw [hA1, hA2, hA3, ← one_smul ℝ (A *ᵥ q), br_smul, vec_eta (A *ᵥ q),
      (frame_brackets _ _ _ _).1, h0] at h
    exact mul_ne_zero hA hq (h.symm.trans (by simp))
  · intro q hq h0
    have h := tbr (p 0) (p 2) (p 3) q
    rw [hA0, hA2, hA3, ← one_smul ℝ (A *ᵥ q), br_smul, vec_eta (A *ᵥ q),
      (frame_brackets _ _ _ _).2.1, h0] at h
    exact mul_ne_zero hA hq (h.symm.trans (by simp))
  · intro q hq h0
    have h := tbr (p 0) (p 1) (p 3) q
    rw [hA0, hA1, hA3, ← one_smul ℝ (A *ᵥ q), br_smul, vec_eta (A *ᵥ q),
      (frame_brackets _ _ _ _).2.2.1, h0] at h
    exact mul_ne_zero hA hq (h.symm.trans (by simp))
  · intro q hq h0
    have h := tbr (p 0) (p 1) (p 2) q
    rw [hA0, hA1, hA2, ← one_smul ℝ (A *ᵥ q), br_smul, vec_eta (A *ᵥ q),
      (frame_brackets _ _ _ _).2.2.2, h0] at h
    exact mul_ne_zero hA hq (h.symm.trans (by simp))

/-- The derivative at 0 of `s ↦ log((a + s b)/(c + s d))` is `b/a - d/c`. -/
theorem hasDerivAt_log_ratio {a b c d : ℝ} (ha : a ≠ 0) (hc : c ≠ 0) :
    HasDerivAt (fun s : ℝ => Real.log ((a + s * b) / (c + s * d))) (b / a - d / c) 0 := by
  have h1 : HasDerivAt (fun s : ℝ => a + s * b) b 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const b).const_add a
  have h2 : HasDerivAt (fun s : ℝ => c + s * d) d 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const d).const_add c
  have h3 := (h1.div h2 (by simpa using hc)).log (by simp [ha, hc])
  convert h3 using 1
  simp only [Pi.div_apply, zero_mul, add_zero]
  field_simp

/-- DΦ(P) v is the derivative of Φ along the line `P + t v`. -/
theorem fderiv_line {ι : Type*} [Fintype ι] (pen : ι → Fin 6 → Fin n)
    (hpen : ∀ i, Function.Injective (pen i)) {P : Config n} (hP : P ∈ Un n) (v : Config n)
    (i : ι) {g' : ℝ} (hg : HasDerivAt (fun t : ℝ => Phi pen (P + t • v) i) g' 0) :
    fderiv ℝ (Phi pen) P v i = g' := by
  have hl : HasDerivAt (fun t : ℝ => P + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add P
  have h := (Phi_differentiableAt pen hpen hP).hasFDerivAt.comp_hasDerivAt_of_eq 0 hl (by simp)
  exact ((hasDerivAt_pi.1 h) i).unique hg

/-- The frame pencils of point `k`: `CR_{01}(2,3;4,k)`, `CR_{02}(1,3;4,k)`, `CR_{12}(0,3;4,k)`
(0-based; `CR_12(3,4;5,k)`, `CR_13(2,4;5,k)`, `CR_23(1,4;5,k)` in the text). -/
def framePencil (hn : 5 ≤ n) (j : Fin 3) (k : Fin n) : Fin 6 → Fin n :=
  ![pt hn (![0, 0, 1] j), pt hn (![1, 2, 2] j), pt hn (![2, 1, 0] j), pt hn 3, pt hn 4, k]

/-- **Theorem 3.7, lower bound.** If the family contains the frame pencils of every point
`k ≥ 5`, then rank DΦ(P) ≥ 3(n - 5). -/
theorem rank_ge (hn : 5 ≤ n) {ι : Type*} [Fintype ι] (pen : ι → Fin 6 → Fin n)
    (hpen : ∀ i, Function.Injective (pen i))
    (hfr : ∀ k : Fin n, 5 ≤ k.val → ∀ j : Fin 3, ∃ i, pen i = framePencil hn j k)
    {P : Config n} (hP : P ∈ Un n) :
    3 * n - 15 ≤ finrank ℝ (LinearMap.range (fderiv ℝ (Phi pen) P : Config n →ₗ[ℝ] ι → ℝ)) := by
  classical
  obtain ⟨A, hA, -, -, hcr0, hcr1, hcr2, hx, hy, hz, hw⟩ := frame_normalise hn hP
  let S := {k : Fin n // 5 ≤ k.val} × Fin 3
  choose idx hidx using fun s : S => hfr s.1.1 s.1.2 s.2
  let D : Config n →ₗ[ℝ] ι → ℝ := (fderiv ℝ (Phi pen) P : Config n →ₗ[ℝ] ι → ℝ)
  let π : (ι → ℝ) →ₗ[ℝ] (S → ℝ) := LinearMap.funLeft ℝ ℝ idx
  let mj : Fin 3 → Fin 4 := ![2, 1, 0]
  have hmj : ∀ j j', mj j = mj j' → j = j' := by decide
  have hmj3 : ∀ j, mj j ≠ 3 := by decide
  have key : ∀ s0 : S, ∃ v, π (D v) = fun s => if s0 = s then 1 else 0 := by
    rintro ⟨⟨k0, hk0⟩, j0⟩
    set a := A *ᵥ P k0 with hadef
    set α : Fin 4 → ℝ := Pi.single (mj j0) (a (mj j0)) with hαdef
    have hAu : A *ᵥ (A⁻¹ *ᵥ α) = α := by
      rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ (isUnit_iff_ne_zero.2 hA),
        Matrix.one_mulVec]
    refine ⟨(Pi.single k0 (A⁻¹ *ᵥ α) : Config n), ?_⟩
    funext s
    obtain ⟨⟨k, hk⟩, j⟩ := s
    change fderiv ℝ (Phi pen) P ((Pi.single k0 (A⁻¹ *ᵥ α) : Config n)) (idx ⟨⟨k, hk⟩, j⟩) = _
    apply fderiv_line pen hpen hP
    have hfun : (fun t : ℝ => Phi pen (P + t • (Pi.single k0 (A⁻¹ *ᵥ α) : Config n)) (idx ⟨⟨k, hk⟩, j⟩))
        = fun t => Real.log (crAt (framePencil hn j k) (P + t • (Pi.single k0 (A⁻¹ *ᵥ α) : Config n))) := by
      funext t; simp only [Phi, hidx]
    rw [hfun]
    have hpt : ∀ x : Fin 5, (pt hn x).val < 5 := fun x => by simp [pt]
    by_cases hkk : k = k0
    · subst hkk
      have hfix : ∀ (t : ℝ) (x : Fin 5),
          (P + t • (Pi.single k (A⁻¹ *ᵥ α) : Config n)) (pt hn x) = P (pt hn x) := by
        intro t x
        have hne : pt hn x ≠ k := fun e => by have := hpt x; rw [e] at this; omega
        simp [Pi.single_eq_of_ne hne]
      have hk' : ∀ t : ℝ, A *ᵥ ((P + t • (Pi.single k (A⁻¹ *ᵥ α) : Config n)) k) = a + t • α := by
        intro t
        simp only [Pi.add_apply, Pi.smul_apply, Pi.single_eq_same]
        rw [Matrix.mulVec_add, Matrix.mulVec_smul, hAu]
      have hkx : ∀ x : Fin 5, pt hn x ≠ k := fun x e => by have := hpt x; rw [e] at this; omega
      have d := fun x y (h : x ≠ y) => pt_ne hn h
      have ha0 : a 0 ≠ 0 := hx _ (hP _ _ _ _ (d 1 2 (by decide)) (d 1 3 (by decide)) (hkx 1)
        (d 2 3 (by decide)) (hkx 2) (hkx 3))
      have ha1 : a 1 ≠ 0 := hy _ (hP _ _ _ _ (d 0 2 (by decide)) (d 0 3 (by decide)) (hkx 0)
        (d 2 3 (by decide)) (hkx 2) (hkx 3))
      have ha2 : a 2 ≠ 0 := hz _ (hP _ _ _ _ (d 0 1 (by decide)) (d 0 3 (by decide)) (hkx 0)
        (d 1 3 (by decide)) (hkx 1) (hkx 3))
      have ha3 : a 3 ≠ 0 := hw _ (hP _ _ _ _ (d 0 1 (by decide)) (d 0 2 (by decide)) (hkx 0)
        (d 1 2 (by decide)) (hkx 1) (hkx 2))
      have ham : ∀ j, a (mj j) ≠ 0 := by
        intro j; fin_cases j
        · exact ha2
        · exact ha1
        · exact ha0
      have hform : (fun t : ℝ =>
          Real.log (crAt (framePencil hn j k) (P + t • (Pi.single k (A⁻¹ *ᵥ α) : Config n))))
          = fun t => Real.log ((a (mj j) + t * α (mj j)) / (a 3 + t * α 3)) := by
        funext t
        congr 1
        have e := hk' t
        set Q := P + t • (Pi.single k (A⁻¹ *ᵥ α) : Config n) with hQ
        fin_cases j
        · change cr (Q (pt hn 0)) (Q (pt hn 1)) (Q (pt hn 2)) (Q (pt hn 3)) (Q (pt hn 4)) (Q k) = _
          rw [show Q (pt hn 0) = P (pt hn 0) from hfix t 0, show Q (pt hn 1) = P (pt hn 1) from hfix t 1,
            show Q (pt hn 2) = P (pt hn 2) from hfix t 2, show Q (pt hn 3) = P (pt hn 3) from hfix t 3,
            show Q (pt hn 4) = P (pt hn 4) from hfix t 4, hcr0, show A *ᵥ Q k = a + t • α from e]
          simp [mj]
        · change cr (Q (pt hn 0)) (Q (pt hn 2)) (Q (pt hn 1)) (Q (pt hn 3)) (Q (pt hn 4)) (Q k) = _
          rw [show Q (pt hn 0) = P (pt hn 0) from hfix t 0, show Q (pt hn 1) = P (pt hn 1) from hfix t 1,
            show Q (pt hn 2) = P (pt hn 2) from hfix t 2, show Q (pt hn 3) = P (pt hn 3) from hfix t 3,
            show Q (pt hn 4) = P (pt hn 4) from hfix t 4, hcr1, show A *ᵥ Q k = a + t • α from e]
          simp [mj]
        · change cr (Q (pt hn 1)) (Q (pt hn 2)) (Q (pt hn 0)) (Q (pt hn 3)) (Q (pt hn 4)) (Q k) = _
          rw [show Q (pt hn 0) = P (pt hn 0) from hfix t 0, show Q (pt hn 1) = P (pt hn 1) from hfix t 1,
            show Q (pt hn 2) = P (pt hn 2) from hfix t 2, show Q (pt hn 3) = P (pt hn 3) from hfix t 3,
            show Q (pt hn 4) = P (pt hn 4) from hfix t 4, hcr2, show A *ᵥ Q k = a + t • α from e]
          simp [mj]
      rw [hform]
      convert hasDerivAt_log_ratio (b := α (mj j)) (d := α 3) (ham j) ha3 using 1
      have hα3 : α 3 = 0 := by
        rw [hαdef, Pi.single_eq_of_ne (hmj3 j0).symm]
      by_cases hj : j0 = j
      · subst hj
        rw [hα3]
        simp [hαdef, ham]
      · have hne : (⟨⟨k, hk0⟩, j0⟩ : S) ≠ ⟨⟨k, hk⟩, j⟩ := fun e => hj (congrArg Prod.snd e)
        have hαj : α (mj j) = 0 := by
          rw [hαdef, Pi.single_eq_of_ne (fun e => hj (hmj _ _ e).symm)]
        simp [hne, hα3, hαj]
    · have hconst : ∀ t : ℝ, crAt (framePencil hn j k) (P + t • (Pi.single k0 (A⁻¹ *ᵥ α) : Config n))
          = crAt (framePencil hn j k) P := by
        intro t
        have hfix : ∀ x : Fin 6, (P + t • (Pi.single k0 (A⁻¹ *ᵥ α) : Config n)) (framePencil hn j k x)
            = P (framePencil hn j k x) := by
          intro x
          have hne : framePencil hn j k x ≠ k0 := by
            intro e
            fin_cases x
            all_goals first
              | exact hkk e
              | (fin_cases j <;> simp [framePencil, pt, Fin.ext_iff] at e <;> omega)
          simp [Pi.single_eq_of_ne hne]
        simp only [crAt, hfix]
      have hne : (⟨⟨k0, hk0⟩, j0⟩ : S) ≠ ⟨⟨k, hk⟩, j⟩ :=
        fun e => hkk (congrArg (fun s : S => s.1.1) e).symm
      simp only [hconst, hne, ite_false]
      exact hasDerivAt_const _ _
  have hsurj : LinearMap.range (π ∘ₗ D) = ⊤ := by
    rw [eq_top_iff]
    intro y _
    rw [pi_eq_sum_univ y]
    refine Submodule.sum_mem _ fun s0 _ => Submodule.smul_mem _ _ ?_
    obtain ⟨v, hv⟩ := key s0
    exact ⟨v, hv⟩
  have hsub : Fintype.card {k : Fin n // 5 ≤ k.val} = n - 5 := by
    have e : {k : Fin n // 5 ≤ k.val} ≃ Fin (n - 5) :=
      { toFun := fun k => ⟨k.1.val - 5, by have := k.1.isLt; have := k.2; omega⟩
        invFun := fun i => ⟨⟨i.val + 5, by have := i.isLt; omega⟩, by simp⟩
        left_inv := fun k => by
          ext; simp only [Fin.val_mk]; have := k.2; omega
        right_inv := fun i => by ext; simp }
    rw [Fintype.card_congr e, Fintype.card_fin]
  have hcard : Fintype.card S = 3 * n - 15 := by
    simp only [S, Fintype.card_prod, hsub, Fintype.card_fin]; omega
  have h1 : finrank ℝ (LinearMap.range (π ∘ₗ D)) = 3 * n - 15 := by
    rw [hsurj, finrank_top, Module.finrank_fintype_fun_eq_card, hcard]
  have h2 : finrank ℝ (LinearMap.range (π ∘ₗ D)) ≤ finrank ℝ (LinearMap.range D) := by
    rw [LinearMap.range_comp]
    exact Submodule.finrank_map_le _ _
  rw [← h1]
  exact h2

/-- **Theorem 3.7 (rank for every n).** For n ≥ 5, P in U_n, and any family of pencils (ordered
6-tuples of distinct points) that contains the frame pencils of every point `k ≥ 5`, the
Jacobian of the pencil map has rank exactly 3n - 15. -/
theorem rank_pencil_map (hn : 5 ≤ n) {ι : Type*} [Fintype ι] (pen : ι → Fin 6 → Fin n)
    (hpen : ∀ i, Function.Injective (pen i))
    (hfr : ∀ k : Fin n, 5 ≤ k.val → ∀ j : Fin 3, ∃ i, pen i = framePencil hn j k)
    {P : Config n} (hP : P ∈ Un n) :
    finrank ℝ (LinearMap.range (fderiv ℝ (Phi pen) P : Config n →ₗ[ℝ] ι → ℝ)) = 3 * n - 15 :=
  le_antisymm (rank_le hn pen hpen hP) (rank_ge hn pen hpen hfr hP)

/-- **Theorem 3.7 over all pencils.** The pencil map over every ordered 6-tuple of distinct
points has Jacobian rank exactly 3n - 15 on U_n. -/
theorem rank_all_pencils (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    finrank ℝ (LinearMap.range (fderiv ℝ (Phi (fun t : {t : Fin 6 → Fin n // Function.Injective t} =>
      t.1)) P : Config n →ₗ[ℝ] {t : Fin 6 → Fin n // Function.Injective t} → ℝ)) = 3 * n - 15 := by
  refine rank_pencil_map hn _ (fun t => t.2) (fun k hk j => ?_) hP
  have hinj : Function.Injective (framePencil hn j k) := by
    intro x y hxy
    fin_cases x <;> fin_cases y <;> fin_cases j <;>
      simp_all [framePencil, pt, Fin.ext_iff] <;> omega
  exact ⟨⟨_, hinj⟩, rfl⟩

end Capri
