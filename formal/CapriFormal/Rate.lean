/-
Proposition 10 (MATHEMATICS.md Proposition 5.5): the rate of κ_J near a configuration with exactly one
zero bracket.

Setting: `P0` has unit points and exactly one zero bracket `[I]` (every bracket of four distinct
points with another index set is nonzero); `P_d` is a curve of unit configurations, continuous
at `d = 0` with `P_0 = P0`, along which `b_I(P_d)` has a nonzero derivative at 0. The curve
`P_d`, the unit normalisation of `P0 + dV`, is such a curve (`normalise_curve`).

* `rate_used`: if `[I]` is used by J1-J3, then `c/d ≤ κ_J(P_d) ≤ C/d` for small `d > 0`.
* `rate_unused`: if every bracket of J1-J3 is nonzero at P0 (in particular if `[I]` is one of
  the four unused brackets), κ_J(P_d) stays bounded near `d = 0`.
-/
import CapriFormal.Kappa

open Filter Topology Matrix Finset

namespace Capri

/-- The signs of the four log-brackets in `log|J_k|`. -/
def sgn : Fin 4 → ℝ := ![1, 1, -1, -1]

theorem partial_logJ (k : Fin 3) {P : Config 6} (hP : ∀ i, bJ k i P ≠ 0) (m : Fin 6) (j : Fin 4) :
    fderiv ℝ (logJ k) P (ec m j) =
      ∑ i, sgn i * (brPart (Jq k i 0) (Jq k i 1) (Jq k i 2) (Jq k i 3) P m j / bJ k i P) := by
  apply fderiv_eq_line (logJ_differentiableAt k hP)
  simp only [logJ, bJ_line]
  convert hasDerivAt_logratio (hP 0) (hP 1) (hP 2) (hP 3) using 1
  simp [sgn, Fin.sum_univ_four]
  ring

/-- Non-axis points: in each `J_k`, every bracket has a point that lies in exactly one other
bracket of `J_k`. -/
theorem nonaxis (k : Fin 3) (i0 : Fin 4) :
    ∃ s i1 : Fin 4, i1 ≠ i0 ∧ (∀ i, i ≠ i0 → i ≠ i1 → ∀ t, Jq k i t ≠ Jq k i0 s) := by
  revert k i0; decide

theorem Jq_tuples_distinct (k : Fin 3) (i i' : Fin 4) (h : i ≠ i') : Jq k i ≠ Jq k i' := by
  revert k i i'; decide

theorem Jq_image_eq (k k' : Fin 3) (i i' : Fin 4)
    (h : Finset.image (Jq k' i') univ = Finset.image (Jq k i) univ) : Jq k' i' = Jq k i := by
  revert k k' i i'; decide

theorem outside_point (k : Fin 3) (i0 : Fin 4) : ∃ z : Fin 6, ∀ t, Jq k i0 t ≠ z := by
  revert k i0; decide

theorem update_ok (k : Fin 3) (i0 s : Fin 4) (z : Fin 6) (hz : ∀ t, Jq k i0 t ≠ z) :
    Function.Injective (Function.update (Jq k i0) s z) ∧
      Finset.image (Function.update (Jq k i0) s z) univ ≠ Finset.image (Jq k i0) univ := by
  revert k i0 s z; decide

theorem brPart_out {a b c d x : Fin 6} (ha : a ≠ x) (hb : b ≠ x) (hc : c ≠ x) (hd : d ≠ x)
    (P : Config 6) (j : Fin 4) : brPart a b c d P x j = 0 := by
  simp [brPart, Ne.symm ha, Ne.symm hb, Ne.symm hc, Ne.symm hd]

theorem br_lin1 (a y c d : Fin 4 → ℝ) : br a y c d = ∑ j, y j * br a (ev j) c d := by
  simp only [br_expand, ev, Fin.sum_univ_four]; simp [Pi.single_apply]; ring
theorem br_lin2 (a b y d : Fin 4 → ℝ) : br a b y d = ∑ j, y j * br a b (ev j) d := by
  simp only [br_expand, ev, Fin.sum_univ_four]; simp [Pi.single_apply]; ring
theorem br_lin3 (a b c y : Fin 4 → ℝ) : br a b c y = ∑ j, y j * br a b c (ev j) := by
  simp only [br_expand, ev, Fin.sum_univ_four]; simp [Pi.single_apply]; ring

/-- Replacing the point in slot `s` of a bracket by `y` gives `∑_j y_j ∂_j`: the cofactor
vector of slot `s` represents the bracket as a linear form in that slot. -/
theorem brf_update (q : Fin 4 → Fin 6) (hq : Function.Injective q) (s : Fin 4) (z : Fin 6)
    (P : Config 6) :
    brf (Function.update q s z 0) (Function.update q s z 1) (Function.update q s z 2)
      (Function.update q s z 3) P
      = ∑ j, P z j * brPart (q 0) (q 1) (q 2) (q 3) P (q s) j := by
  have d := fun a b (h : a ≠ b) => hq.ne h
  fin_cases s <;>
  · simp only [brf, brPart, Function.update, Fin.isValue]
    simp [d 0 1 (by decide), d 0 2 (by decide), d 0 3 (by decide), d 1 2 (by decide),
      d 1 3 (by decide), d 2 3 (by decide), (d 0 1 (by decide)).symm, (d 0 2 (by decide)).symm,
      (d 0 3 (by decide)).symm, (d 1 2 (by decide)).symm, (d 1 3 (by decide)).symm,
      (d 2 3 (by decide)).symm]
    first
      | exact br_lin0 _ _ _ _
      | exact br_lin1 _ _ _ _
      | exact br_lin2 _ _ _ _
      | exact br_lin3 _ _ _ _

theorem brPart_continuous (a b c d x : Fin 6) (j : Fin 4) :
    Continuous fun P : Config 6 => brPart a b c d P x j := by
  unfold brPart
  split_ifs <;> (try simp only [br_expand]) <;> fun_prop

theorem ev_norm (j : Fin 4) : ‖WithLp.toLp 2 (ev j)‖ = 1 := by
  rw [EuclideanSpace.norm_eq, Real.sqrt_eq_one]
  fin_cases j <;> simp [ev, Fin.sum_univ_four, Pi.single_apply]

/-- A cofactor entry of a unit configuration is at most 1 in absolute value (Hadamard). -/
theorem brPart_le_one {a b c d : Fin 6} (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) {P : Config 6}
    (hunit : ∀ m, ‖WithLp.toLp 2 (P m)‖ = 1) (x : Fin 6) (j : Fin 4) :
    |brPart a b c d P x j| ≤ 1 := by
  have H : ∀ u v w y : Fin 4 → ℝ, ‖WithLp.toLp 2 u‖ = 1 → ‖WithLp.toLp 2 v‖ = 1 →
      ‖WithLp.toLp 2 w‖ = 1 → ‖WithLp.toLp 2 y‖ = 1 → |br u v w y| ≤ 1 := by
    intro u v w y hu hv hw hy
    have := br_hadamard u v w y
    rwa [hu, hv, hw, hy, mul_one, mul_one, mul_one] at this
  unfold brPart
  by_cases h1 : x = a
  · subst h1; simp [hab, hac, had]; exact H _ _ _ _ (ev_norm j) (hunit _) (hunit _) (hunit _)
  by_cases h2 : x = b
  · subst h2; simp [h1, hbc, hbd]; exact H _ _ _ _ (hunit _) (ev_norm j) (hunit _) (hunit _)
  by_cases h3 : x = c
  · subst h3; simp [h1, h2, hcd]; exact H _ _ _ _ (hunit _) (hunit _) (ev_norm j) (hunit _)
  by_cases h4 : x = d
  · subst h4; simp [h1, h2, h3]; exact H _ _ _ _ (hunit _) (hunit _) (hunit _) (ev_norm j)
  simp [h1, h2, h3, h4]

theorem Jq_injective (k : Fin 3) (i : Fin 4) : Function.Injective (Jq k i) := by
  revert k i; decide

theorem eventually_abs_ge {f : ℝ → ℝ} (hf : ContinuousAt f 0) (h0 : f 0 ≠ 0) :
    ∀ᶠ d in 𝓝 0, |f 0| / 2 ≤ |f d| := by
  have hc : ContinuousAt (fun d => |f d|) 0 := hf.abs
  have hpos : |f 0| / 2 < |f 0| := by have := abs_pos.2 h0; linarith
  filter_upwards [hc.eventually (lt_mem_nhds hpos)] with d hd using hd.le

/-- A function vanishing at 0 with derivative `b' ≠ 0` there lies between `|b'||d|/2` and
`2|b'||d|` near 0. -/
theorem deriv_bounds {f : ℝ → ℝ} {b' : ℝ} (hf : HasDerivAt f b' 0) (h0 : f 0 = 0)
    (hb' : b' ≠ 0) : ∀ᶠ d in 𝓝 0, |b'| * |d| / 2 ≤ |f d| ∧ |f d| ≤ 2 * |b'| * |d| := by
  have h := (hasDerivAt_iff_isLittleO.1 hf).def (c := |b'| / 2) (by positivity)
  filter_upwards [h] with d hd
  simp only [h0, sub_zero, smul_eq_mul, Real.norm_eq_abs] at hd
  have e : |d * b'| = |b'| * |d| := by rw [abs_mul, mul_comm]
  constructor
  · have := abs_sub_abs_le_abs_sub (d * b') (f d)
    rw [abs_sub_comm] at this
    nlinarith [abs_nonneg d, abs_nonneg b']
  · have := abs_sub_abs_le_abs_sub (f d) (d * b')
    nlinarith [abs_nonneg d, abs_nonneg b']

theorem bJ_contAt (k : Fin 3) (i : Fin 4) {Pd : ℝ → Config 6} (hcont : ContinuousAt Pd 0) :
    ContinuousAt (fun d => bJ k i (Pd d)) 0 :=
  (brf_differentiable _ _ _ _).continuous.continuousAt.comp hcont

/-- **Proposition 10, unused case.** If every bracket of J1-J3 is nonzero at P0, κ_J stays
bounded along any continuous curve of unit configurations through P0. -/
theorem rate_unused {P0 : Config 6} {Pd : ℝ → Config 6} (hcont : ContinuousAt Pd 0)
    (h0 : Pd 0 = P0) (hunit : ∀ᶠ d in 𝓝 0, ∀ m, ‖WithLp.toLp 2 (Pd d m)‖ = 1)
    (hall : ∀ k i, bJ k i P0 ≠ 0) : ∃ C, ∀ᶠ d in 𝓝 0, kappaJ (Pd d) ≤ C := by
  set μ := (univ.inf' univ_nonempty fun ki : Fin 3 × Fin 4 => |bJ ki.1 ki.2 P0|) / 2 with hμdef
  have hμ : 0 < μ := by
    have : 0 < univ.inf' univ_nonempty fun ki : Fin 3 × Fin 4 => |bJ ki.1 ki.2 P0| :=
      (Finset.lt_inf'_iff _).2 fun ki _ => abs_pos.2 (hall ki.1 ki.2)
    positivity
  have hnear : ∀ᶠ d in 𝓝 0, ∀ ki : Fin 3 × Fin 4, μ ≤ |bJ ki.1 ki.2 (Pd d)| := by
    refine Filter.eventually_all.2 fun ki => ?_
    have h := eventually_abs_ge (bJ_contAt ki.1 ki.2 hcont) (by rw [h0]; exact hall ki.1 ki.2)
    filter_upwards [h] with d hd
    have hle : univ.inf' univ_nonempty (fun ki : Fin 3 × Fin 4 => |bJ ki.1 ki.2 P0|)
        ≤ |bJ ki.1 ki.2 P0| := Finset.inf'_le _ (mem_univ ki)
    rw [h0] at hd
    linarith [hμdef]
  refine ⟨8 / μ, ?_⟩
  filter_upwards [hnear, hunit] with d hd hu
  have hδ : μ ≤ deltaJ (Pd d) := Finset.le_inf' _ _ fun ki _ => hd ki
  exact le_trans (kappa_le hu (lt_of_lt_of_le hμ hδ))
    (div_le_div_of_nonneg_left (by norm_num) hμ hδ)

/-- **Proposition 10, used case.** Let P0 have exactly one zero bracket, `[I]` = bracket `i0` of
`J_k`, and let `P_d` be a continuous curve of unit configurations through P0 along which
`b_I(P_d)` has a nonzero derivative at 0. Then `c/d ≤ κ_J(P_d) ≤ C/d` for small `d > 0`. -/
theorem rate_used (k : Fin 3) (i0 : Fin 4) {P0 : Config 6} {Pd : ℝ → Config 6}
    (hcont : ContinuousAt Pd 0) (h0 : Pd 0 = P0)
    (hunit : ∀ᶠ d in 𝓝 0, ∀ m, ‖WithLp.toLp 2 (Pd d m)‖ = 1)
    (hone : ∀ q : Fin 4 → Fin 6, Function.Injective q →
      Finset.image q univ ≠ Finset.image (Jq k i0) univ → brf (q 0) (q 1) (q 2) (q 3) P0 ≠ 0)
    (hzero : bJ k i0 P0 = 0)
    {b' : ℝ} (hder : HasDerivAt (fun d => bJ k i0 (Pd d)) b' 0) (hb' : b' ≠ 0) :
    ∃ c C : ℝ, 0 < c ∧ 0 < C ∧
      ∀ᶠ d in 𝓝[>] 0, c / d ≤ kappaJ (Pd d) ∧ kappaJ (Pd d) ≤ C / d := by
  classical
  have hb'pos : 0 < |b'| := abs_pos.2 hb'
  -- the other brackets of J1-J3 are nonzero at P0
  have hother : ∀ k' i', Jq k' i' ≠ Jq k i0 → bJ k' i' P0 ≠ 0 := fun k' i' hne =>
    hone _ (Jq_injective k' i') (fun h => hne (Jq_image_eq k k' i0 i' h))
  have hsame : ∀ k' i', Jq k' i' = Jq k i0 → ∀ P, bJ k' i' P = bJ k i0 P := by
    intro k' i' h P; simp only [bJ, h]
  -- a uniform lower bound μ for the other brackets near 0
  set f : Fin 3 × Fin 4 → ℝ := fun ki =>
    if Jq ki.1 ki.2 = Jq k i0 then 1 else |bJ ki.1 ki.2 P0| with hfdef
  have hfpos : ∀ ki, 0 < f ki := by
    intro ki; simp only [hfdef]; split_ifs with h
    · norm_num
    · exact abs_pos.2 (hother _ _ h)
  set μ := (univ.inf' univ_nonempty f) / 2 with hμdef
  have hμ : 0 < μ := by
    have : 0 < univ.inf' univ_nonempty f := (Finset.lt_inf'_iff _).2 fun ki _ => hfpos ki
    positivity
  have hnear : ∀ᶠ d in 𝓝 0, ∀ ki : Fin 3 × Fin 4, Jq ki.1 ki.2 ≠ Jq k i0 →
      μ ≤ |bJ ki.1 ki.2 (Pd d)| := by
    refine Filter.eventually_all.2 fun ki => ?_
    by_cases hki : Jq ki.1 ki.2 = Jq k i0
    · exact Filter.Eventually.of_forall fun d h => absurd hki h
    · have h := eventually_abs_ge (bJ_contAt ki.1 ki.2 hcont)
        (by rw [h0]; exact hother _ _ hki)
      filter_upwards [h] with d hd _
      have hle : univ.inf' univ_nonempty f ≤ f ki := Finset.inf'_le _ (mem_univ ki)
      simp only [hfdef, hki, ite_false] at hle
      rw [h0] at hd
      linarith [hμdef]
  have hI := deriv_bounds hder (by simp only [h0]; exact hzero) hb'
  -- the non-axis point `x` of `[I]` and its cofactor
  obtain ⟨s, i1, hi1, hout⟩ := nonaxis k i0
  obtain ⟨z, hz⟩ := outside_point k i0
  obtain ⟨hinjU, himU⟩ := update_ok k i0 s z hz
  have hcof : ∃ j, brPart (Jq k i0 0) (Jq k i0 1) (Jq k i0 2) (Jq k i0 3) P0 (Jq k i0 s) j ≠ 0 := by
    by_contra hall
    push Not at hall
    have h := hone _ hinjU himU
    rw [brf_update (Jq k i0) (Jq_injective k i0) s z P0] at h
    simp [hall] at h
  obtain ⟨jx, hjx⟩ := hcof
  set u0 := |brPart (Jq k i0 0) (Jq k i0 1) (Jq k i0 2) (Jq k i0 3) P0 (Jq k i0 s) jx|
  have hu0 : 0 < u0 := abs_pos.2 hjx
  have hu : ∀ᶠ d in 𝓝 0, u0 / 2 ≤
      |brPart (Jq k i0 0) (Jq k i0 1) (Jq k i0 2) (Jq k i0 3) (Pd d) (Jq k i0 s) jx| := by
    have h := eventually_abs_ge (f := fun d =>
      brPart (Jq k i0 0) (Jq k i0 1) (Jq k i0 2) (Jq k i0 3) (Pd d) (Jq k i0 s) jx)
      ((brPart_continuous _ _ _ _ _ _).continuousAt.comp hcont) (by simp only [h0]; exact hjx)
    simpa only [h0] using h
  refine ⟨u0 / (6 * |b'|), 16 / |b'|, by positivity, by positivity, ?_⟩
  have hsmall : ∀ᶠ d in 𝓝[>] (0 : ℝ), 0 < d ∧ d < μ / |b'| ∧ d < u0 * μ / (12 * |b'|) := by
    have h0' : ∀ᶠ d in 𝓝[>] (0 : ℝ), 0 < d := self_mem_nhdsWithin
    refine h0'.and ?_
    have h1 : ∀ᶠ d in 𝓝 (0 : ℝ), d < μ / |b'| := eventually_lt_nhds (by positivity)
    have h2 : ∀ᶠ d in 𝓝 (0 : ℝ), d < u0 * μ / (12 * |b'|) := eventually_lt_nhds (by positivity)
    exact (h1.and h2).filter_mono nhdsWithin_le_nhds
  filter_upwards [hI.filter_mono nhdsWithin_le_nhds, hnear.filter_mono nhdsWithin_le_nhds,
    hu.filter_mono nhdsWithin_le_nhds, hunit.filter_mono nhdsWithin_le_nhds, hsmall]
    with d hId hnd hud hun ⟨hd0, hd1, hd2⟩
  have habsd : |d| = d := abs_of_pos hd0
  rw [habsd] at hId
  obtain ⟨hIlo, hIhi⟩ := hId
  have hIpos : 0 < |bJ k i0 (Pd d)| := lt_of_lt_of_le (by positivity) hIlo
  -- every bracket of J1-J3 is at least |b'| d / 2
  have hlow : ∀ ki : Fin 3 × Fin 4, |b'| * d / 2 ≤ |bJ ki.1 ki.2 (Pd d)| := by
    intro ki
    by_cases hki : Jq ki.1 ki.2 = Jq k i0
    · rw [hsame _ _ hki]; exact hIlo
    · have := hnd ki hki
      have : |b'| * d < μ := by rwa [lt_div_iff₀ hb'pos, mul_comm] at hd1
      linarith
  have hne : ∀ k' i', bJ k' i' (Pd d) ≠ 0 := fun k' i' h => by
    have := hlow (k', i'); rw [h, abs_zero] at this
    have : 0 < |b'| * d / 2 := by positivity
    linarith
  constructor
  · -- lower bound through the coordinate (x, jx) of ∇ log|J_k|
    have hcomp : |fderiv ℝ (logJ k) (Pd d) (ec (Jq k i0 s) jx)| ≤ kappaJ (Pd d) := by
      have h1 : |fderiv ℝ (logJ k) (Pd d) (ec (Jq k i0 s) jx)| ≤ ‖grad (logJ k) (Pd d)‖ := by
        have := PiLp.norm_apply_le (grad (logJ k) (Pd d)) (Jq k i0 s, jx)
        simpa [grad, Real.norm_eq_abs] using this
      exact le_trans h1 (Finset.le_sup' (fun k => ‖grad (logJ k) (Pd d)‖) (mem_univ k))
    refine le_trans ?_ hcomp
    rw [partial_logJ k (hne k) (Jq k i0 s) jx]
    rw [Finset.sum_eq_add_of_mem i0 i1 (mem_univ _) (mem_univ _) (Ne.symm hi1) (fun c _ hc => by
      have hzero' : brPart (Jq k c 0) (Jq k c 1) (Jq k c 2) (Jq k c 3) (Pd d) (Jq k i0 s) jx = 0 :=
        brPart_out (hout c hc.1 hc.2 0) (hout c hc.1 hc.2 1) (hout c hc.1 hc.2 2)
          (hout c hc.1 hc.2 3) _ _
      rw [hzero', zero_div, mul_zero])]
    set A := brPart (Jq k i0 0) (Jq k i0 1) (Jq k i0 2) (Jq k i0 3) (Pd d) (Jq k i0 s) jx
    set B := brPart (Jq k i1 0) (Jq k i1 1) (Jq k i1 2) (Jq k i1 3) (Pd d) (Jq k i0 s) jx
    have hB : |B| ≤ 1 := by
      obtain ⟨h1, h2, h3, h4, h5, h6⟩ := Jq_distinct k i1
      exact brPart_le_one h1 h2 h3 h4 h5 h6 hun _ _
    have hb1 : μ ≤ |bJ k i1 (Pd d)| := hnd (k, i1) (Jq_tuples_distinct k i1 i0 hi1)
    have hsg : ∀ i, |sgn i| = 1 := by intro i; fin_cases i <;> simp [sgn]
    have hA : u0 / 2 / (2 * |b'| * d) ≤ |A / bJ k i0 (Pd d)| := by
      rw [abs_div]
      exact div_le_div₀ (abs_nonneg _) hud hIpos hIhi
    have hBb : |B / bJ k i1 (Pd d)| ≤ 1 / μ := by
      rw [abs_div]; exact div_le_div₀ zero_le_one hB hμ hb1
    have htri : |A / bJ k i0 (Pd d)| - |B / bJ k i1 (Pd d)| ≤
        |sgn i0 * (A / bJ k i0 (Pd d)) + sgn i1 * (B / bJ k i1 (Pd d))| := by
      have := abs_sub_abs_le_abs_sub (sgn i0 * (A / bJ k i0 (Pd d)))
        (-(sgn i1 * (B / bJ k i1 (Pd d))))
      rw [abs_mul, hsg, one_mul, abs_neg, abs_mul, hsg, one_mul, sub_neg_eq_add] at this
      exact this
    have hfinal : u0 / (6 * |b'|) / d ≤ u0 / 2 / (2 * |b'| * d) - 1 / μ := by
      have e1 : u0 / 2 / (2 * |b'| * d) = u0 / (4 * |b'| * d) := by field_simp; ring
      have e2 : u0 / (6 * |b'|) / d = u0 / (6 * |b'| * d) := by field_simp
      rw [e1, e2]
      have : 1 / μ ≤ u0 / (12 * |b'| * d) := by
        rw [div_le_div_iff₀ hμ (by positivity)]
        rw [lt_div_iff₀ (by positivity)] at hd2
        nlinarith
      have e3 : u0 / (4 * |b'| * d) - u0 / (6 * |b'| * d) = u0 / (12 * |b'| * d) := by
        field_simp; ring
      linarith
    linarith
  · -- upper bound: κ_J ≤ 8/δ_J and δ_J ≥ |b'| d / 2
    have hδ : |b'| * d / 2 ≤ deltaJ (Pd d) := Finset.le_inf' _ _ fun ki _ => hlow ki
    have hδpos : 0 < |b'| * d / 2 := by positivity
    calc kappaJ (Pd d) ≤ 8 / deltaJ (Pd d) := kappa_le hun (lt_of_lt_of_le hδpos hδ)
      _ ≤ 8 / (|b'| * d / 2) := div_le_div_of_nonneg_left (by norm_num) hδpos hδ
      _ = 16 / |b'| / d := by field_simp; ring

/-- The unit normalisation `P_d = (P0 + d V)/‖P0 + d V‖` of the text. -/
noncomputable def normalise (P0 V : Config 6) (d : ℝ) : Config 6 :=
  fun m => ‖WithLp.toLp 2 (P0 m + d • V m)‖⁻¹ • (P0 m + d • V m)

/-- The text's `P_d` is a continuous curve of unit configurations through `P0`. -/
theorem normalise_curve (P0 V : Config 6) (hunit : ∀ m, ‖WithLp.toLp 2 (P0 m)‖ = 1) :
    ContinuousAt (normalise P0 V) 0 ∧ normalise P0 V 0 = P0 ∧
      ∀ᶠ d in 𝓝 0, ∀ m, ‖WithLp.toLp 2 (normalise P0 V d m)‖ = 1 := by
  have hn : ∀ m, Continuous fun d : ℝ => ‖WithLp.toLp 2 (P0 m + d • V m)‖ := fun m =>
    (PiLp.continuous_toLp 2 _ |>.comp (by fun_prop)).norm
  refine ⟨?_, ?_, ?_⟩
  · refine continuousAt_pi.2 fun m => ?_
    refine ContinuousAt.smul ((hn m).continuousAt.inv₀ (by simp [hunit m])) (by fun_prop)
  · funext m; simp [normalise, hunit m]
  · refine Filter.eventually_all.2 fun m => ?_
    filter_upwards [(hn m).continuousAt.eventually_ne (by simp [hunit m] : ‖WithLp.toLp 2
      (P0 m + (0 : ℝ) • V m)‖ ≠ 0)] with d hd
    simp only [normalise, WithLp.toLp_smul, norm_smul, norm_inv, norm_norm]
    exact inv_mul_cancel₀ hd

end Capri
