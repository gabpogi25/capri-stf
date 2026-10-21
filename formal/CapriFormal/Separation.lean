/-
Corollary 3.8 (separation).

* `separation_signed` (b): for P, Q in U_n, the 3(n - 5) signed frame cross-ratios agree if and
  only if Q is in the gauge orbit of P.
* `Un_isOpen`: U_n is open.
* `separation_local` (a): near P in U_n, a configuration whose frame values `log|CR|` equal
  those of P is in the gauge orbit of P; conversely every configuration in the orbit has the
  same pencil map (`Phi_act`).

Scope: (a) is proved as orbit membership; that the gauge can be taken near the identity is not
formalized.
-/
import CapriFormal.RankFull

open Matrix Filter Topology

namespace Capri

variable {n : ℕ}

/-- The gauge orbit of `P`. -/
def orbit (P : Config n) : Set (Config n) :=
  {Q | ∃ (A : Matrix (Fin 4) (Fin 4) ℝ) (l : Fin n → ℝ), A.det ≠ 0 ∧ (∀ k, l k ≠ 0) ∧
    Q = act A l P}

theorem crAt_act (t : Fin 6 → Fin n) (A : Matrix (Fin 4) (Fin 4) ℝ) (hA : A.det ≠ 0)
    (l : Fin n → ℝ) (hl : ∀ k, l k ≠ 0) (P : Config n) : crAt t (act A l P) = crAt t P := by
  simp only [crAt, act]
  rw [cr_gauge A hA _ _ _ _ _ _ (hl _) (hl _) (hl _) (hl _) (hl _) (hl _)]

/-- Two vectors with all four coordinates nonzero and equal coordinate ratios are proportional. -/
theorem prop_of_ratios {a b : Fin 4 → ℝ} (ha3 : a 3 ≠ 0) (hb3 : b 3 ≠ 0)
    (h0 : a 0 / a 3 = b 0 / b 3) (h1 : a 1 / a 3 = b 1 / b 3) (h2 : a 2 / a 3 = b 2 / b 3) :
    b = (b 3 / a 3) • a := by
  field_simp at h0 h1 h2
  have hc : ∀ m : Fin 4, b m = b 3 / a 3 * a m := by
    intro m
    fin_cases m
    · show b 0 = b 3 / a 3 * a 0
      rw [div_mul_eq_mul_div, eq_div_iff ha3]; linarith
    · show b 1 = b 3 / a 3 * a 1
      rw [div_mul_eq_mul_div, eq_div_iff ha3]; linarith
    · show b 2 = b 3 / a 3 * a 2
      rw [div_mul_eq_mul_div, eq_div_iff ha3]; linarith
    · show b 3 = b 3 / a 3 * a 3
      rw [div_mul_cancel₀ _ ha3]
  funext m
  simp only [Pi.smul_apply, smul_eq_mul, hc m]

/-- **Corollary 3.8 (b), signed separation.** For P, Q in U_n (n ≥ 5), the signed frame
cross-ratios of every point k ≥ 5 agree if and only if Q is in the gauge orbit of P. -/
theorem separation_signed (hn : 5 ≤ n) {P Q : Config n} (hP : P ∈ Un n) (hQ : Q ∈ Un n) :
    (∀ k : Fin n, 5 ≤ k.val → ∀ j : Fin 3,
      crAt (framePencil hn j k) Q = crAt (framePencil hn j k) P) ↔ Q ∈ orbit P := by
  constructor
  · intro h
    obtain ⟨A, hA, ⟨s, hs, hAs⟩, hA4, hcr0, hcr1, hcr2, -, -, -, hw⟩ := frame_normalise hn hP
    obtain ⟨B, hB, ⟨t, ht, hBt⟩, hB4, hdr0, hdr1, hdr2, -, -, -, hw'⟩ := frame_normalise hn hQ
    have hkx : ∀ (k : Fin n), 5 ≤ k.val → ∀ x : Fin 5, pt hn x ≠ k := fun k hk x e => by
      have : (pt hn x).val < 5 := by simp [pt]
      rw [e] at this; omega
    have d := fun x y (h : x ≠ y) => pt_ne hn h
    -- the proportionality factors
    have key : ∀ k : Fin n, ∃ l : ℝ, l ≠ 0 ∧ B *ᵥ Q k = l • (A *ᵥ P k) := by
      intro k
      by_cases hk : 5 ≤ k.val
      · have hk' := hkx k hk
        have hw0 : (A *ᵥ P k) 3 ≠ 0 := hw _ (hP _ _ _ _ (d 0 1 (by decide)) (d 0 2 (by decide))
          (hk' 0) (d 1 2 (by decide)) (hk' 1) (hk' 2))
        have hw1 : (B *ᵥ Q k) 3 ≠ 0 := hw' _ (hQ _ _ _ _ (d 0 1 (by decide)) (d 0 2 (by decide))
          (hk' 0) (d 1 2 (by decide)) (hk' 1) (hk' 2))
        have e0 := h k hk 0
        have e1 := h k hk 1
        have e2 := h k hk 2
        change cr (Q (pt hn 0)) (Q (pt hn 1)) (Q (pt hn 2)) (Q (pt hn 3)) (Q (pt hn 4)) (Q k)
          = cr (P (pt hn 0)) (P (pt hn 1)) (P (pt hn 2)) (P (pt hn 3)) (P (pt hn 4)) (P k) at e0
        change cr (Q (pt hn 0)) (Q (pt hn 2)) (Q (pt hn 1)) (Q (pt hn 3)) (Q (pt hn 4)) (Q k)
          = cr (P (pt hn 0)) (P (pt hn 2)) (P (pt hn 1)) (P (pt hn 3)) (P (pt hn 4)) (P k) at e1
        change cr (Q (pt hn 1)) (Q (pt hn 2)) (Q (pt hn 0)) (Q (pt hn 3)) (Q (pt hn 4)) (Q k)
          = cr (P (pt hn 1)) (P (pt hn 2)) (P (pt hn 0)) (P (pt hn 3)) (P (pt hn 4)) (P k) at e2
        rw [hcr0, hdr0] at e0
        rw [hcr1, hdr1] at e1
        rw [hcr2, hdr2] at e2
        refine ⟨(B *ᵥ Q k) 3 / (A *ᵥ P k) 3, div_ne_zero hw1 hw0, ?_⟩
        exact prop_of_ratios hw0 hw1 e2.symm e1.symm e0.symm
      · by_cases h4k : k.val = 4
        · have hkk : k = pt hn 4 := Fin.ext (by simp [pt, h4k])
          rw [hkk]
          exact ⟨1, one_ne_zero, by rw [hA4, hB4, one_smul]⟩
        · set i : Fin 4 := ⟨k.val, by omega⟩ with hidef
          have hkk : k = pt hn i.castSucc := Fin.ext (by simp [pt, i])
          rw [hkk]
          refine ⟨t i / s i, div_ne_zero (ht i) (hs i), ?_⟩
          rw [hAs, hBt, smul_smul, div_mul_cancel₀ _ (hs i)]
    choose l hl0 hl using key
    have hBinv : B⁻¹ * B = 1 := Matrix.nonsing_inv_mul B (isUnit_iff_ne_zero.2 hB)
    refine ⟨B⁻¹ * A, l, ?_, hl0, ?_⟩
    · rw [Matrix.det_mul, Matrix.det_nonsing_inv, Ring.inverse_eq_inv']
      exact mul_ne_zero (inv_ne_zero hB) hA
    · funext k
      have h1 : B⁻¹ *ᵥ (B *ᵥ Q k) = Q k := by
        rw [Matrix.mulVec_mulVec, hBinv, Matrix.one_mulVec]
      simp only [act]
      rw [← h1, hl k, Matrix.mulVec_smul, Matrix.mulVec_mulVec]
  · rintro ⟨A, l, hA, hl, rfl⟩ k _ j
    exact crAt_act _ A hA l hl P

/-- U_n is open. -/
theorem Un_isOpen : IsOpen (Un n) := by
  have h : Un n = ⋂ (a : Fin n) (b : Fin n) (c : Fin n) (d : Fin n),
      {P : Config n | a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
        br (P a) (P b) (P c) (P d) ≠ 0} := by
    ext P; simp [Un]
  rw [h]
  refine isOpen_iInter_of_finite fun a => isOpen_iInter_of_finite fun b =>
    isOpen_iInter_of_finite fun c => isOpen_iInter_of_finite fun d => ?_
  by_cases hd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  · obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hd
    have : {P : Config n | a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
        br (P a) (P b) (P c) (P d) ≠ 0} = {P | br (P a) (P b) (P c) (P d) ≠ 0} := by
      ext P; simp [h1, h2, h3, h4, h5, h6]
    rw [this]
    exact isOpen_ne_fun (br_differentiable a b c d).continuous continuous_const
  · have : {P : Config n | a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
        br (P a) (P b) (P c) (P d) ≠ 0} = Set.univ := by
      ext P
      simp only [Set.mem_setOf_eq, Set.mem_univ, iff_true]
      intro h1 h2 h3 h4 h5 h6
      exact absurd ⟨h1, h2, h3, h4, h5, h6⟩ hd
    rw [this]
    exact isOpen_univ

/-- **Corollary 3.8 (a), local separation with absolute values.** For P in U_n there is a
neighbourhood of P in which every configuration whose frame values `log|CR|` equal those of P is
in the gauge orbit of P. (Conversely, the orbit has the same pencil map: `Phi_act`.) -/
theorem separation_local (hn : 5 ≤ n) {P : Config n} (hP : P ∈ Un n) :
    ∀ᶠ Q in 𝓝 P, (∀ k : Fin n, 5 ≤ k.val → ∀ j : Fin 3,
      Real.log (crAt (framePencil hn j k) Q) = Real.log (crAt (framePencil hn j k) P)) →
      Q ∈ orbit P := by
  have hinj : ∀ (k : Fin n), 5 ≤ k.val → ∀ j : Fin 3, Function.Injective (framePencil hn j k) := by
    intro k hk j x y hxy
    fin_cases x <;> fin_cases y <;> fin_cases j <;>
      simp_all [framePencil, pt, Fin.ext_iff] <;> omega
  -- every frame cross-ratio keeps its sign near P
  have hsign : ∀ k : Fin n, ∀ j : Fin 3, ∀ᶠ Q in 𝓝 P, 5 ≤ k.val →
      0 < crAt (framePencil hn j k) Q * crAt (framePencil hn j k) P := by
    intro k j
    by_cases hk : 5 ≤ k.val
    · have hc := (crAt_differentiableAt (hinj k hk j) hP).continuousAt
      have hne := crAt_ne_zero (hinj k hk j) hP
      have hpos : 0 < crAt (framePencil hn j k) P * crAt (framePencil hn j k) P :=
        mul_self_pos.2 hne
      have : ContinuousAt (fun Q => crAt (framePencil hn j k) Q * crAt (framePencil hn j k) P) P :=
        hc.mul continuousAt_const
      filter_upwards [this.eventually (lt_mem_nhds hpos)] with Q hQ _ using hQ
    · exact Filter.Eventually.of_forall fun Q h => absurd h hk
  have hall : ∀ᶠ Q in 𝓝 P, ∀ k : Fin n, ∀ j : Fin 3, 5 ≤ k.val →
      0 < crAt (framePencil hn j k) Q * crAt (framePencil hn j k) P :=
    Filter.eventually_all.2 fun k => Filter.eventually_all.2 fun j => hsign k j
  filter_upwards [hall, Un_isOpen.mem_nhds hP] with Q hQs hQU hlog
  refine (separation_signed hn hP hQU).1 fun k hk j => ?_
  have hpos := hQs k j hk
  have hneP := crAt_ne_zero (hinj k hk j) hP
  have hneQ : crAt (framePencil hn j k) Q ≠ 0 := fun h0 => by simp [h0] at hpos
  have habs : |crAt (framePencil hn j k) Q| = |crAt (framePencil hn j k) P| := by
    have h := hlog k hk j
    rw [← Real.log_abs, ← Real.log_abs (crAt _ P)] at h
    exact Real.log_injOn_pos (abs_pos.2 hneQ) (abs_pos.2 hneP) h
  rcases abs_eq_abs.1 habs with h | h
  · exact h
  · rw [h] at hpos; nlinarith [mul_self_pos.2 hneP]

end Capri
