/-
The Taylor step of Proposition 5 (MATHEMATICS.md Proposition 4.4): on the event that the noise is
bounded by β, the standardised discrepancy

  z_t = D_t / (m₀ (‖g_t(A)‖ + ‖g_t(B)‖)),   m₀ = μ r,

with the gradients taken at the measured unit representatives `A = unit(q + r ũ)`,
`B = unit(q + r ṽ)`, is within `C r` of `L_t = |g_t(q)·w| / (2 μ ‖g_t(q)‖)`, `w = ũ - ṽ`,
uniformly in `q` over a compact set of unit configurations on which the brackets of `t` are
nonzero and `‖g_t‖ ≥ γ > 0` (`pivotal_taylor`).
-/
import CapriFormal.Taylor
import CapriFormal.Kappa

open Filter Topology Metric

namespace Capri

variable {n : ℕ}

/-- `φ_t = log|CR_t|`. -/
noncomputable def phi (t : Fin 6 → Fin n) : Config n → ℝ := fun Q => Real.log (crAt t Q)

/-- The unit representatives of a configuration. -/
noncomputable def unitize (Q : Config n) : Config n := fun k => ‖WithLp.toLp 2 (Q k)‖⁻¹ • Q k

/-- The standardised discrepancy of pencil `t`. -/
noncomputable def zstat (t : Fin 6 → Fin n) (μ r : ℝ) (q u v : Config n) : ℝ :=
  |phi t (q + r • u) - phi t (q + r • v)| /
    (μ * r * (‖grad (phi t) (unitize (q + r • u))‖ + ‖grad (phi t) (unitize (q + r • v))‖))

/-- Its first-order limit `L_t`. -/
noncomputable def Lstat (t : Fin 6 → Fin n) (μ : ℝ) (q u v : Config n) : ℝ :=
  |fderiv ℝ (phi t) q (u - v)| / (2 * μ * ‖grad (phi t) q‖)

theorem ec_norm_le (k : Fin n) (j : Fin 4) : ‖ec k j‖ ≤ 1 := by
  refine (pi_norm_le_iff_of_nonneg zero_le_one).2 fun m => ?_
  refine (pi_norm_le_iff_of_nonneg zero_le_one).2 fun i => ?_
  by_cases hm : m = k
  · subst hm; by_cases hi : i = j
    · subst hi; simp [ec, ev]
    · simp [ec, ev, Pi.single_eq_of_ne hi]
  · simp [ec, Pi.single_eq_of_ne hm]

/-- The Euclidean gradient is controlled by the operator norm of the derivative. -/
theorem grad_norm_le (L : Config n →L[ℝ] ℝ) :
    ‖(WithLp.toLp 2 (fun kj : Fin n × Fin 4 => L (ec kj.1 kj.2)) : EuclideanSpace ℝ (Fin n × Fin 4))‖
      ≤ Real.sqrt (4 * n) * ‖L‖ := by
  rw [EuclideanSpace.norm_eq]
  have h : ∑ kj : Fin n × Fin 4, ‖L (ec kj.1 kj.2)‖ ^ 2 ≤ (4 * n) * ‖L‖ ^ 2 := by
    calc ∑ kj : Fin n × Fin 4, ‖L (ec kj.1 kj.2)‖ ^ 2 ≤ ∑ _kj : Fin n × Fin 4, ‖L‖ ^ 2 := by
          refine Finset.sum_le_sum fun kj _ => ?_
          have h1 : ‖L (ec kj.1 kj.2)‖ ≤ ‖L‖ := by
            have := L.le_opNorm (ec kj.1 kj.2)
            have h2 := mul_le_mul_of_nonneg_left (ec_norm_le kj.1 kj.2) (norm_nonneg L)
            rw [mul_one] at h2
            linarith
          exact pow_le_pow_left₀ (norm_nonneg _) h1 2
      _ = (4 * n) * ‖L‖ ^ 2 := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
            Fintype.card_fin, nsmul_eq_mul]
          push_cast; ring
  change Real.sqrt (∑ kj : Fin n × Fin 4, ‖L (ec kj.1 kj.2)‖ ^ 2) ≤ _
  calc Real.sqrt (∑ kj : Fin n × Fin 4, ‖L (ec kj.1 kj.2)‖ ^ 2)
      ≤ Real.sqrt ((4 * n) * ‖L‖ ^ 2) := Real.sqrt_le_sqrt h
    _ = Real.sqrt (4 * n) * ‖L‖ := by
        rw [Real.sqrt_mul (by positivity), Real.sqrt_sq (norm_nonneg _)]

theorem grad_eq (f : Config n → ℝ) (P : Config n) :
    grad f P = WithLp.toLp 2 (fun kj : Fin n × Fin 4 => fderiv ℝ f P (ec kj.1 kj.2)) := rfl

theorem grad_sub_le (f : Config n → ℝ) (x y : Config n) :
    ‖grad f x - grad f y‖ ≤ Real.sqrt (4 * n) * ‖fderiv ℝ f x - fderiv ℝ f y‖ := by
  have e : grad f x - grad f y =
      WithLp.toLp 2 (fun kj : Fin n × Fin 4 => (fderiv ℝ f x - fderiv ℝ f y) (ec kj.1 kj.2)) := by
    ext kj; simp [grad]
  rw [e]
  exact grad_norm_le (fderiv ℝ f x - fderiv ℝ f y)

/-- Distance from a unit vector to the unit normalisation of a perturbation of it. -/
theorem unit_dist (q h : Fin 4 → ℝ) (hq : ‖WithLp.toLp 2 q‖ = 1) :
    ‖WithLp.toLp 2 (‖WithLp.toLp 2 (q + h)‖⁻¹ • (q + h)) - WithLp.toLp 2 q‖
      ≤ 2 * ‖WithLp.toLp 2 h‖ := by
  set v := WithLp.toLp 2 (q + h) with hv
  have hvq : v = WithLp.toLp 2 q + WithLp.toLp 2 h := by simp [hv]
  have e : WithLp.toLp 2 (‖v‖⁻¹ • (q + h)) = ‖v‖⁻¹ • v := by simp [hv]
  rw [e]
  by_cases h0 : ‖v‖ = 0
  · rw [h0, inv_zero, zero_smul, zero_sub, norm_neg, hq]
    have : ‖WithLp.toLp 2 h‖ = 1 := by
      have hv0 : v = 0 := norm_eq_zero.1 h0
      have : WithLp.toLp 2 h = -WithLp.toLp 2 q := by
        rw [hvq] at hv0; exact eq_neg_of_add_eq_zero_right hv0
      rw [this, norm_neg, hq]
    rw [this]; norm_num
  · have h1 : ‖‖v‖⁻¹ • v - v‖ = |1 - ‖v‖| := by
      have : ‖v‖⁻¹ • v - v = (‖v‖⁻¹ - 1) • v := by rw [sub_smul, one_smul]
      rw [this, norm_smul, Real.norm_eq_abs]
      rw [show (‖v‖⁻¹ - 1) = (1 - ‖v‖) / ‖v‖ by field_simp, abs_div, abs_norm,
        div_mul_cancel₀ _ h0]
    have h2 : |1 - ‖v‖| ≤ ‖WithLp.toLp 2 h‖ := by
      rw [← hq, hvq]
      have := abs_norm_sub_norm_le (WithLp.toLp 2 q) (WithLp.toLp 2 q + WithLp.toLp 2 h)
      simpa using this
    have h3 : ‖v - WithLp.toLp 2 q‖ = ‖WithLp.toLp 2 h‖ := by rw [hvq]; simp
    calc ‖‖v‖⁻¹ • v - WithLp.toLp 2 q‖ = ‖(‖v‖⁻¹ • v - v) + (v - WithLp.toLp 2 q)‖ := by
          congr 1; abel
      _ ≤ ‖‖v‖⁻¹ • v - v‖ + ‖v - WithLp.toLp 2 q‖ := norm_add_le _ _
      _ ≤ 2 * ‖WithLp.toLp 2 h‖ := by rw [h1, h3]; linarith

theorem sup_le_euclid (x : Fin 4 → ℝ) : ‖x‖ ≤ ‖WithLp.toLp 2 x‖ := by
  refine (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2 fun i => ?_
  have := PiLp.norm_apply_le (WithLp.toLp 2 x) i
  simpa using this

theorem euclid_le_sup (x : Fin 4 → ℝ) : ‖WithLp.toLp 2 x‖ ≤ 2 * ‖x‖ := by
  rw [EuclideanSpace.norm_eq]
  have h : ∑ i, ‖(WithLp.toLp 2 x) i‖ ^ 2 ≤ 4 * ‖x‖ ^ 2 := by
    calc ∑ i, ‖(WithLp.toLp 2 x) i‖ ^ 2 ≤ ∑ _i : Fin 4, ‖x‖ ^ 2 := by
          refine Finset.sum_le_sum fun i _ => pow_le_pow_left₀ (norm_nonneg _) ?_ 2
          simpa using norm_le_pi_norm x i
      _ = 4 * ‖x‖ ^ 2 := by simp
  calc Real.sqrt _ ≤ Real.sqrt (4 * ‖x‖ ^ 2) := Real.sqrt_le_sqrt h
    _ = 2 * ‖x‖ := by
      rw [Real.sqrt_mul (by norm_num), Real.sqrt_sq (norm_nonneg _),
        show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]

/-- For a unit configuration `q`, `‖unit(q + h) - q‖ ≤ 4 ‖h‖` (sup norms). -/
theorem unitize_dist (q h : Config n) (hq : ∀ k, ‖WithLp.toLp 2 (q k)‖ = 1) :
    ‖unitize (q + h) - q‖ ≤ 4 * ‖h‖ := by
  refine (pi_norm_le_iff_of_nonneg (by positivity)).2 fun k => ?_
  have h1 := unit_dist (q k) (h k) (hq k)
  have h2 := sup_le_euclid ((unitize (q + h) - q) k)
  have h3 := euclid_le_sup (h k)
  have h4 : ‖h k‖ ≤ ‖h‖ := norm_le_pi_norm h k
  have e : WithLp.toLp 2 ((unitize (q + h) - q) k) =
      WithLp.toLp 2 (‖WithLp.toLp 2 (q k + h k)‖⁻¹ • (q k + h k)) - WithLp.toLp 2 (q k) := by
    simp [unitize]
  rw [e] at h2
  linarith

theorem unitize_of_unit (q : Config n) (hq : ∀ k, ‖WithLp.toLp 2 (q k)‖ = 1) : unitize q = q := by
  funext k; simp [unitize, hq k]

/-- **Proposition 5, the Taylor step.** On a compact set `K` of unit configurations where the
brackets of `t` are nonzero and `‖g_t‖ ≥ γ > 0`, for noise bounded by `β` (sup norm) and
`0 < r ≤ r₀`: `|z_t - L_t| ≤ C r`, uniformly in `q ∈ K`. -/
theorem pivotal_taylor (t : Fin 6 → Fin n) {K : Set (Config n)} (hK : IsCompact K)
    (hKU : K ⊆ Ut t) (hKunit : ∀ q ∈ K, ∀ k, ‖WithLp.toLp 2 (q k)‖ = 1)
    {γ μ β : ℝ} (hγ : 0 < γ) (hgrad : ∀ q ∈ K, γ ≤ ‖grad (phi t) q‖) (hμ : 0 < μ)
    (hβ : 0 ≤ β) :
    ∃ C r₀ : ℝ, 0 ≤ C ∧ 0 < r₀ ∧ ∀ q ∈ K, ∀ u v : Config n, ‖u‖ ≤ β → ‖v‖ ≤ β →
      ∀ r, 0 < r → r ≤ r₀ → |zstat t μ r q u v - Lstat t μ q u v| ≤ C * r := by
  obtain ⟨C1, r1, hC1, hr1, hsf⟩ := scale_family t hK hKU hβ
  obtain ⟨M, δ2, hM, hδ2, hlip⟩ := fderiv_lip_uniform (Ut_isOpen t) hK hKU (logcr_contDiffOn t)
  have hcontD : ContinuousOn (fun q => ‖fderiv ℝ (phi t) q‖) K := by
    have h := ((logcr_contDiffOn t).continuousOn_fderiv_of_isOpen (Ut_isOpen t)
      (by norm_num)).mono hKU
    exact ContinuousOn.norm (f := fderiv ℝ (phi t)) h
  obtain ⟨Mq0, hMq0⟩ := hK.exists_bound_of_continuousOn hcontD
  set Mq := max Mq0 0
  have hMq : ∀ q ∈ K, ‖fderiv ℝ (phi t) q‖ ≤ Mq := fun q hq => by
    have := hMq0 q hq
    rw [Real.norm_eq_abs] at this
    exact le_trans (le_trans (le_abs_self _) this) (le_max_left _ _)
  set Lg := Real.sqrt (4 * n) * M
  set ρ := 4 * β
  have hLg : 0 ≤ Lg := by positivity
  set C := C1 / (μ * γ) + 2 * Mq * β * Lg * ρ / (μ * γ ^ 2)
  refine ⟨C, min r1 (min (δ2 / (ρ + 1)) (γ / (2 * (Lg * ρ + 1)))), by positivity,
    lt_min hr1 (lt_min (by positivity) (by positivity)), ?_⟩
  intro q hq u v hu hv r hr hrr
  have hr1' : r ≤ r1 := le_trans hrr (min_le_left _ _)
  have hr2 : r ≤ δ2 / (ρ + 1) := le_trans hrr (le_trans (min_le_right _ _) (min_le_left _ _))
  have hr3 : r ≤ γ / (2 * (Lg * ρ + 1)) :=
    le_trans hrr (le_trans (min_le_right _ _) (min_le_right _ _))
  -- the gradients at the measured unit representatives
  have hdist : ∀ w : Config n, ‖w‖ ≤ β → ‖unitize (q + r • w) - q‖ ≤ ρ * r := by
    intro w hw
    refine le_trans (unitize_dist q (r • w) (hKunit q hq)) ?_
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos hr]
    nlinarith
  have hgd : ∀ w : Config n, ‖w‖ ≤ β →
      |‖grad (phi t) (unitize (q + r • w))‖ - ‖grad (phi t) q‖| ≤ Lg * ρ * r := by
    intro w hw
    have hd := hdist w hw
    have hd2 : ‖unitize (q + r • w) - q‖ ≤ δ2 := by
      refine le_trans hd ?_
      rw [le_div_iff₀ (by positivity)] at hr2
      nlinarith
    calc _ ≤ ‖grad (phi t) (unitize (q + r • w)) - grad (phi t) q‖ := abs_norm_sub_norm_le _ _
      _ ≤ Real.sqrt (4 * n) * ‖fderiv ℝ (phi t) (unitize (q + r • w)) - fderiv ℝ (phi t) q‖ :=
          grad_sub_le _ _ _
      _ ≤ Real.sqrt (4 * n) * (M * ‖unitize (q + r • w) - q‖) :=
          mul_le_mul_of_nonneg_left (hlip q hq _ hd2) (by positivity)
      _ ≤ Lg * ρ * r := by
          have := mul_le_mul_of_nonneg_left hd (by positivity : 0 ≤ Real.sqrt (4 * n) * M)
          simp only [Lg]; nlinarith
  have hsmall : Lg * ρ * r ≤ γ / 2 := by
    rw [le_div_iff₀ (by positivity)] at hr3
    nlinarith
  set g := ‖grad (phi t) q‖
  set GA := ‖grad (phi t) (unitize (q + r • u))‖
  set GB := ‖grad (phi t) (unitize (q + r • v))‖
  have hg : γ ≤ g := hgrad q hq
  have hgA := hgd u hu
  have hgB := hgd v hv
  have hS : g ≤ GA + GB := by
    have := (abs_le.1 hgA).1; have := (abs_le.1 hgB).1; linarith
  have hSd : |2 * g - (GA + GB)| ≤ 2 * (Lg * ρ * r) := by
    rw [abs_le]; constructor
    · have := (abs_le.1 hgA).2; have := (abs_le.1 hgB).2; linarith
    · have := (abs_le.1 hgA).1; have := (abs_le.1 hgB).1; linarith
  -- the first-order expansion of D
  set D := |phi t (q + r • u) - phi t (q + r • v)|
  set a := |fderiv ℝ (phi t) q (u - v)|
  have hD : |D - r * a| ≤ C1 * r ^ 2 := by
    have h := hsf q hq u v hu hv r hr.le hr1' 1 1 (fun _ => 1) (fun _ => 1) (by simp) (by simp)
      (fun _ => one_ne_zero) (fun _ => one_ne_zero)
    have e : ∀ Q : Config n, act 1 (fun _ => 1) Q = Q := fun Q => by funext k; simp [act]
    rw [e, e] at h
    exact h
  have ha : a ≤ Mq * (2 * β) := by
    calc a ≤ ‖fderiv ℝ (phi t) q‖ * ‖u - v‖ := by
          have := (fderiv ℝ (phi t) q).le_opNorm (u - v)
          rwa [Real.norm_eq_abs] at this
      _ ≤ Mq * (2 * β) := by
          refine mul_le_mul (hMq q hq) ?_ (norm_nonneg _) (le_max_right _ _)
          have := norm_sub_le u v; linarith
  have hgpos : 0 < g := lt_of_lt_of_le hγ hg
  have hSpos : 0 < GA + GB := lt_of_lt_of_le hgpos hS
  -- the algebra
  have hμ0 : μ ≠ 0 := hμ.ne'
  have hr0 : r ≠ 0 := hr.ne'
  have hg0 : g ≠ 0 := hgpos.ne'
  have hS0 : GA + GB ≠ 0 := hSpos.ne'
  have e : zstat t μ r q u v - Lstat t μ q u v =
      (2 * g * (D - r * a) + r * a * (2 * g - (GA + GB))) / (2 * μ * r * g * (GA + GB)) := by
    change D / (μ * r * (GA + GB)) - a / (2 * μ * g) = _
    field_simp
    ring
  have hdpos : 0 < 2 * μ * r * g * (GA + GB) := mul_pos (by positivity) hSpos
  rw [e, abs_div, abs_of_pos hdpos]
  have hN : |2 * g * (D - r * a) + r * a * (2 * g - (GA + GB))|
      ≤ 2 * g * (C1 * r ^ 2) + r * (Mq * (2 * β)) * (2 * (Lg * ρ * r)) := by
    calc _ ≤ |2 * g * (D - r * a)| + |r * a * (2 * g - (GA + GB))| := abs_add_le _ _
      _ = 2 * g * |D - r * a| + r * a * |2 * g - (GA + GB)| := by
          rw [abs_mul (2 * g) (D - r * a), abs_mul (r * a) (2 * g - (GA + GB)),
            abs_of_pos (by positivity : (0:ℝ) < 2 * g),
            abs_of_nonneg (mul_nonneg hr.le (abs_nonneg _))]
      _ ≤ _ := by
          gcongr
  have hden : 2 * μ * r * g * g ≤ 2 * μ * r * g * (GA + GB) :=
    mul_le_mul_of_nonneg_left hS (by positivity)
  calc _ ≤ (2 * g * (C1 * r ^ 2) + r * (Mq * (2 * β)) * (2 * (Lg * ρ * r))) /
        (2 * μ * r * g * g) := div_le_div₀ (by positivity) hN (by positivity) hden
    _ = r * (C1 / (μ * g) + 2 * Mq * β * Lg * ρ / (μ * g ^ 2)) := by
        field_simp
    _ ≤ r * C := by
        refine mul_le_mul_of_nonneg_left (add_le_add ?_ ?_) hr.le
        · exact div_le_div_of_nonneg_left hC1 (by positivity) (mul_le_mul_of_nonneg_left hg hμ.le)
        · exact div_le_div_of_nonneg_left (by positivity) (by positivity)
            (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hγ.le hg 2) hμ.le)
    _ = C * r := mul_comm _ _

end Capri
