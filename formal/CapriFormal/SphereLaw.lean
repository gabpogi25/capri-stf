/-
Fact (i) of Proposition 9: for `x` uniform on S³ and a unit vector `n`,
`P(|⟨n, x⟩| < a) = Ψ(a) = (2/π)(a √(1 - a²) + arcsin a)` for `0 < a ≤ 1`.

The proof computes the volume of the cone `{y : 0 < ‖y‖ < 1, |y₀| < a ‖y‖}` in ℝ⁴ by splitting
ℝ⁴ = ℝ¹ × ℝ³ (Fubini): each slice is a spherical shell in ℝ³, and the radial integral is
elementary (`integral_cube_sqrt`).
-/
import CapriFormal.Sphere
import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

open MeasureTheory Metric Set Real intervalIntegral
open scoped RealInnerProductSpace ENNReal

namespace Capri

/-- `Ψ(a) = (2/π)(a √(1 - a²) + arcsin a)`. -/
noncomputable def Psi (a : ℝ) : ℝ := 2 / π * (a * √(1 - a ^ 2) + arcsin a)

/-- The antiderivative of `(1 - y²)^{3/2}`. -/
noncomputable def F32 (y : ℝ) : ℝ := y * √(1 - y ^ 2) ^ 3 / 4 + 3 / 8 * (arcsin y + y * √(1 - y ^ 2))

theorem hasDerivAt_F32 {y : ℝ} (h0 : -1 < y) (h1 : y < 1) :
    HasDerivAt F32 (√(1 - y ^ 2) ^ 3) y := by
  have hpos : 0 < 1 - y ^ 2 := by nlinarith
  have hs : HasDerivAt (fun y : ℝ => √(1 - y ^ 2)) (-y / √(1 - y ^ 2)) y := by
    have h := ((hasDerivAt_pow 2 y).const_sub 1).sqrt hpos.ne'
    convert h using 1; field_simp; ring
  have ha := hasDerivAt_arcsin h0.ne' h1.ne
  have h := (((hasDerivAt_id y).mul (hs.pow 3)).div_const 4).add
    ((ha.add ((hasDerivAt_id y).mul hs)).const_mul (3 / 8))
  have hsq : √(1 - y ^ 2) ^ 2 = 1 - y ^ 2 := Real.sq_sqrt hpos.le
  have hsp : 0 < √(1 - y ^ 2) := Real.sqrt_pos.2 hpos
  convert h using 1
  · funext x; simp only [F32, Pi.add_apply, Pi.mul_apply, Pi.pow_apply, id_eq]
  · simp only [Pi.pow_apply, id_eq]
    norm_num
    field_simp
    have h4 : √(1 - y ^ 2) ^ 4 = (1 - y ^ 2) ^ 2 := by
      rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, hsq]
    rw [h4, hsq]
    ring

theorem F32_zero : F32 0 = 0 := by simp [F32]

/-- `∫₀ᵃ (1 - y²)^{3/2} dy = F32 a` for `0 ≤ a ≤ 1`. -/
theorem integral_cube_sqrt {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    ∫ y in (0 : ℝ)..a, √(1 - y ^ 2) ^ 3 = F32 a := by
  rw [integral_eq_sub_of_hasDerivAt_of_le ha0 (by unfold F32; fun_prop)
    (fun y hy => hasDerivAt_F32 (by linarith [hy.1]) (by linarith [hy.2]))
    ((by fun_prop : Continuous fun y : ℝ => √(1 - y ^ 2) ^ 3).intervalIntegrable _ _),
    F32_zero, sub_zero]

/-- Splitting `ℝ^{n+m} = ℝ^n × ℝ^m` as Euclidean spaces. -/
noncomputable def split (n m : ℕ) : EuclideanSpace ℝ (Fin (n + m)) ≃ₗᵢ[ℝ]
    WithLp 2 (EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)) :=
  (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ finSumFinEquiv.symm).trans
    (PiLp.sumPiLpEquivProdLpPiLp (𝕜 := ℝ) 2 (fun _ => ℝ))

/-- The inverse of the splitting, on the plain product. -/
noncomputable def joinv (n m : ℕ) (p : EuclideanSpace ℝ (Fin n) × EuclideanSpace ℝ (Fin m)) :
    EuclideanSpace ℝ (Fin (n + m)) := (split n m).symm (WithLp.toLp 2 p)

theorem joinv_mp (n m : ℕ) : MeasurePreserving (joinv n m) volume volume :=
  (split n m).symm.measurePreserving.comp (WithLp.volume_preserving_toLp _ _)

theorem joinv_measurable (n m : ℕ) : Measurable (joinv n m) := (joinv_mp n m).measurable

theorem joinv_norm_sq (n m : ℕ) (u : EuclideanSpace ℝ (Fin n)) (v : EuclideanSpace ℝ (Fin m)) :
    ‖joinv n m (u, v)‖ ^ 2 = ‖u‖ ^ 2 + ‖v‖ ^ 2 := by
  rw [joinv, LinearIsometryEquiv.norm_map, WithLp.prod_norm_sq_eq_of_L2]; simp

theorem joinv_fst (n m : ℕ) (u : EuclideanSpace ℝ (Fin n)) (v : EuclideanSpace ℝ (Fin m))
    (i : Fin n) : joinv n m (u, v) (Fin.castAdd m i) = u i := by
  simp [joinv, split]

theorem joinv_snd (n m : ℕ) (u : EuclideanSpace ℝ (Fin n)) (v : EuclideanSpace ℝ (Fin m))
    (j : Fin m) : joinv n m (u, v) (Fin.natAdd n j) = v j := by
  simp [joinv, split]

/-- **Fubini for the splitting.** -/
theorem vol_split (n m : ℕ) {S : Set (EuclideanSpace ℝ (Fin (n + m)))} (hS : MeasurableSet S) :
    volume S = ∫⁻ u, volume {v | joinv n m (u, v) ∈ S} := by
  rw [← (joinv_mp n m).measure_preimage hS.nullMeasurableSet, Measure.volume_eq_prod,
    Measure.prod_apply (hS.preimage (joinv_measurable n m))]
  rfl

theorem norm_fin1 (u : EuclideanSpace ℝ (Fin 1)) : ‖u‖ = |u 0| := by
  rw [EuclideanSpace.norm_eq, Fin.sum_univ_one, Real.norm_eq_abs, sq_abs, Real.sqrt_sq_eq_abs]

theorem vol_ball_fin1 : (volume : Measure (EuclideanSpace ℝ (Fin 1))).real (ball 0 1) = 2 := by
  rw [measureReal_def, EuclideanSpace.volume_ball]
  simp only [Fintype.card_fin, ENNReal.ofReal_one, one_mul, pow_one, Nat.cast_one]
  rw [show (1 : ℝ) / 2 + 1 = 1 / 2 + 1 from rfl, Real.Gamma_add_one (by norm_num),
    Real.Gamma_one_half_eq, ENNReal.toReal_ofReal (by positivity)]
  have : 0 < √π := Real.sqrt_pos.2 Real.pi_pos
  field_simp

/-- The inequalities behind the slice of the cone of fact (i). -/
theorem ineq_i {a s w N : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) (hs : 0 ≤ s) (hw : 0 ≤ w)
    (hN0 : 0 ≤ N) (hN : N ^ 2 = s ^ 2 + w ^ 2) :
    (0 < N ∧ N < 1 ∧ s < a * N) ↔ (w < √(1 - s ^ 2) ∧ √(1 - a ^ 2) / a * s < w) := by
  have hc0 : 0 ≤ √(1 - a ^ 2) / a := by positivity
  have hca : (√(1 - a ^ 2) / a) ^ 2 * a ^ 2 = 1 - a ^ 2 := by
    rw [div_pow, Real.sq_sqrt (by nlinarith)]; field_simp
  set c := √(1 - a ^ 2) / a
  constructor
  · rintro ⟨h0, h1, h2⟩
    refine ⟨?_, ?_⟩
    · rw [Real.lt_sqrt hw]; nlinarith
    · have hs2 : s ^ 2 < a ^ 2 * N ^ 2 := by
        have := mul_self_lt_mul_self hs h2; nlinarith
      have key : (c * s) ^ 2 * a ^ 2 < w ^ 2 * a ^ 2 := by nlinarith
      have key' : (c * s) ^ 2 < w ^ 2 := lt_of_mul_lt_mul_right key (by positivity)
      exact (pow_lt_pow_iff_left₀ (by positivity) hw two_ne_zero).1 key'
  · rintro ⟨h1, h2⟩
    rw [Real.lt_sqrt hw] at h1
    have hwpos : 0 < w := lt_of_le_of_lt (by positivity) h2
    have hcw : (c * s) ^ 2 < w ^ 2 := pow_lt_pow_left₀ h2 (by positivity) two_ne_zero
    refine ⟨?_, ?_, ?_⟩
    · by_contra h; push Not at h
      have : N = 0 := le_antisymm h hN0
      rw [this] at hN; nlinarith
    · nlinarith
    · have : s ^ 2 < (a * N) ^ 2 := by nlinarith
      exact (pow_lt_pow_iff_left₀ hs (by positivity) two_ne_zero).1 this

/-- Volume of a spherical shell in ℝ³. -/
theorem vol_shell (r₁ r₂ : ℝ) (hr₁ : 0 ≤ r₁) :
    volume (ball (0 : EuclideanSpace ℝ (Fin 3)) r₂ \ closedBall 0 r₁)
      = ENNReal.ofReal (π * 4 / 3 * max (r₂ ^ 3 - r₁ ^ 3) 0) := by
  rcases lt_or_ge r₁ r₂ with h | h
  · have hr₂ : 0 < r₂ := lt_of_le_of_lt hr₁ h
    rw [measure_sdiff (closedBall_subset_ball h) measurableSet_closedBall.nullMeasurableSet
      measure_closedBall_lt_top.ne, EuclideanSpace.volume_ball_fin_three, EuclideanSpace.volume_closedBall_fin_three,
      ← ENNReal.ofReal_pow hr₂.le, ← ENNReal.ofReal_pow hr₁, ← ENNReal.ofReal_mul (by positivity),
      ← ENNReal.ofReal_mul (by positivity), ← ENNReal.ofReal_sub _ (by positivity)]
    congr 1
    have : r₁ ^ 3 < r₂ ^ 3 := pow_lt_pow_left₀ h hr₁ (by norm_num)
    rw [max_eq_left (by linarith)]; ring
  · have hsub : ball (0 : EuclideanSpace ℝ (Fin 3)) r₂ \ closedBall 0 r₁ = ∅ := by
      rw [sdiff_eq_empty]; exact (ball_subset_closedBall).trans (closedBall_subset_closedBall h)
    rw [hsub, measure_empty, max_eq_right, mul_zero, ENNReal.ofReal_zero]
    nlinarith [sq_nonneg (r₂ + r₁ / 2), sq_nonneg r₁]

/-- The cone of fact (i). -/
def coneI (a : ℝ) : Set E4 := {y | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ |y 0| < a * ‖y‖}

theorem coneI_measurable (a : ℝ) : MeasurableSet (coneI a) := by
  have h1 : Continuous fun y : E4 => ‖y‖ := continuous_norm
  have h2 : Continuous fun y : E4 => |y 0| := by fun_prop
  exact (measurableSet_lt measurable_const h1.measurable).inter
    ((measurableSet_lt h1.measurable measurable_const).inter
      (measurableSet_lt h2.measurable (h1.measurable.const_mul a)))

/-- The radial profile of the cone of fact (i). -/
noncomputable def GI (a s : ℝ) : ℝ := π * 4 / 3 * max (√(1 - s ^ 2) ^ 3 - (√(1 - a ^ 2) / a * s) ^ 3) 0

theorem slice_I {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) (u : EuclideanSpace ℝ (Fin 1)) :
    volume {v : EuclideanSpace ℝ (Fin 3) | joinv 1 3 (u, v) ∈ coneI a}
      = ENNReal.ofReal (GI a ‖u‖) := by
  have hset : {v : EuclideanSpace ℝ (Fin 3) | joinv 1 3 (u, v) ∈ coneI a}
      = ball 0 (√(1 - ‖u‖ ^ 2)) \ closedBall 0 (√(1 - a ^ 2) / a * ‖u‖) := by
    ext v
    have h0 : joinv 1 3 (u, v) 0 = u 0 := joinv_fst 1 3 u v 0
    simp only [coneI, mem_ofPred_eq, mem_sdiff, mem_ball_zero_iff, mem_closedBall_zero_iff, not_le]
    rw [h0, ← norm_fin1]
    exact ineq_i ha0 ha1 (norm_nonneg _) (norm_nonneg _) (norm_nonneg _) (joinv_norm_sq 1 3 u v)
  rw [hset, vol_shell _ _ (by positivity), GI, mul_pow]

theorem GI_zero {a s : ℝ} (ha0 : 0 < a) (hs : a ≤ s) : GI a s = 0 := by
  have hr : 0 ≤ √(1 - a ^ 2) := Real.sqrt_nonneg _
  have h1 : √(1 - s ^ 2) ≤ √(1 - a ^ 2) := Real.sqrt_le_sqrt (by nlinarith)
  have h2 : √(1 - a ^ 2) ≤ √(1 - a ^ 2) / a * s := by
    rw [div_mul_eq_mul_div, le_div_iff₀ ha0]; nlinarith
  have h3 : √(1 - s ^ 2) ^ 3 ≤ (√(1 - a ^ 2) / a * s) ^ 3 :=
    pow_le_pow_left₀ (Real.sqrt_nonneg _) (h1.trans h2) 3
  rw [GI, max_eq_right (by linarith), mul_zero]

theorem GI_eq {a s : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) (hs0 : 0 ≤ s) (hs : s ≤ a) :
    GI a s = π * 4 / 3 * (√(1 - s ^ 2) ^ 3 - (√(1 - a ^ 2) / a) ^ 3 * s ^ 3) := by
  have hca : (√(1 - a ^ 2) / a) ^ 2 * a ^ 2 = 1 - a ^ 2 := by
    rw [div_pow, Real.sq_sqrt (by nlinarith)]; field_simp
  have hc0 : 0 ≤ √(1 - a ^ 2) / a := by positivity
  have hle : √(1 - a ^ 2) / a * s ≤ √(1 - s ^ 2) := by
    apply Real.le_sqrt_of_sq_le
    have : (√(1 - a ^ 2) / a * s) ^ 2 * a ^ 2 ≤ (1 - s ^ 2) * a ^ 2 := by
      have hs2 : s ^ 2 ≤ a ^ 2 := pow_le_pow_left₀ hs0 hs 2
      nlinarith
    exact le_of_mul_le_mul_right this (by positivity)
  have h3 := pow_le_pow_left₀ (by positivity) hle 3
  rw [GI, max_eq_left (by linarith), mul_pow]

theorem GI_nonneg (a s : ℝ) : 0 ≤ GI a s := by
  unfold GI; have := Real.pi_pos; positivity

theorem GI_continuous (a : ℝ) : Continuous (GI a) := by
  unfold GI; fun_prop

/-- **Volume of the cone of fact (i).** -/
theorem vol_coneI {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    volume (coneI a) = ENNReal.ofReal (π * (arcsin a + a * √(1 - a ^ 2))) := by
  have hcont : Continuous fun u : EuclideanSpace ℝ (Fin 1) => GI a ‖u‖ :=
    (GI_continuous a).comp continuous_norm
  have hsupp : HasCompactSupport fun u : EuclideanSpace ℝ (Fin 1) => GI a ‖u‖ :=
    HasCompactSupport.intro (isCompact_closedBall 0 a) fun u hu => by
      rw [mem_closedBall_zero_iff, not_le] at hu; exact GI_zero ha0 hu.le
  have hint := hcont.integrable_of_hasCompactSupport hsupp (μ := volume)
  have hmeas := coneI_measurable a
  calc volume (coneI a) = ∫⁻ u, volume {v | joinv 1 3 (u, v) ∈ coneI a} := vol_split 1 3 hmeas
    _ = ∫⁻ u, ENNReal.ofReal (GI a ‖u‖) := lintegral_congr fun u => slice_I ha0 ha1 u
    _ = ENNReal.ofReal (∫ u, GI a ‖u‖) :=
        (ofReal_integral_eq_lintegral_ofReal hint (ae_of_all _ fun u => GI_nonneg a _)).symm
    _ = _ := by
      congr 1
      rw [integral_fun_norm_addHaar volume (GI a)]
      simp only [finrank_euclideanSpace, Fintype.card_fin, vol_ball_fin1, Nat.sub_self, pow_zero,
        one_smul, smul_eq_mul, one_mul]
      rw [setIntegral_eq_of_subset_of_forall_sdiff_eq_zero measurableSet_Ioi Ioc_subset_Ioi_self
        (fun x hx => GI_zero ha0 (by
          simp only [mem_sdiff, mem_Ioi, mem_Ioc, not_and, not_le] at hx
          exact (hx.2 hx.1).le)),
        ← intervalIntegral.integral_of_le ha0.le]
      rw [intervalIntegral.integral_congr (g := fun s => π * 4 / 3 *
          (√(1 - s ^ 2) ^ 3 - (√(1 - a ^ 2) / a) ^ 3 * s ^ 3)) (fun s hs => by
        rw [uIcc_of_le ha0.le] at hs; exact GI_eq ha0 ha1 hs.1 hs.2)]
      rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_sub
        ((by fun_prop : Continuous fun y : ℝ => √(1 - y ^ 2) ^ 3).intervalIntegrable _ _)
        ((by fun_prop : Continuous fun y : ℝ => (√(1 - a ^ 2) / a) ^ 3 * y ^ 3).intervalIntegrable _ _),
        intervalIntegral.integral_const_mul, integral_pow, integral_cube_sqrt ha0.le ha1]
      have hc : (√(1 - a ^ 2) / a) ^ 3 * a ^ 4 = a * √(1 - a ^ 2) ^ 3 := by
        field_simp
      unfold F32
      norm_num
      linear_combination (-(2 * π / 3)) * hc

/-- **Fact (i) of Proposition 9, coordinate form.** For `x` uniform on S³ and `0 < a ≤ 1`,
`P(|x₀| < a) = Ψ(a)`. -/
theorem unif_abs_lt_e0 {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    unifS3 {x : S3 | |(x : E4) 0| < a} = ENNReal.ofReal (Psi a) := by
  set P : E4 → Prop := fun y => |y 0| < a * ‖y‖
  have hP : ScaleInv P := fun c hc y => by
    simp only [P, PiLp.smul_apply, smul_eq_mul, abs_mul, norm_smul, Real.norm_eq_abs,
      abs_of_pos hc]
    constructor
    · intro h; nlinarith
    · intro h; nlinarith
  have hset : {x : S3 | |(x : E4) 0| < a} = {x : S3 | P x} := by
    ext x; simp only [P, mem_ofPred_eq]
    rw [show ‖(x : E4)‖ = 1 from norm_eq_of_mem_sphere x, mul_one]
  have hm : MeasurableSet {x : S3 | P x} := by
    have h1 : Continuous fun x : S3 => |(x : E4) 0| := by fun_prop
    have h2 : Continuous fun x : S3 => a * ‖(x : E4)‖ := by fun_prop
    exact measurableSet_lt h1.measurable h2.measurable
  rw [hset, unifS3_cone hP hm]
  change volume (coneI a) / _ = _
  rw [vol_coneI ha0 ha1, ← ENNReal.ofReal_div_of_pos (by positivity), Psi]
  congr 1
  have := Real.pi_pos
  field_simp; ring

/-- **Fact (i) of Proposition 9.** For `x` uniform on S³, a unit vector `n` and `0 < a ≤ 1`,
`P(|⟨n, x⟩| < a) = Ψ(a)`. -/
theorem unif_abs_inner_lt {n : E4} (hn : ‖n‖ = 1) {a : ℝ} (ha0 : 0 < a) (ha1 : a ≤ 1) :
    unifS3 {x : S3 | |⟪n, (x : E4)⟫| < a} = ENNReal.ofReal (Psi a) := by
  obtain ⟨R, hR⟩ := rot_to_e0 hn
  set P : E4 → Prop := fun y => |y 0| < a * ‖y‖
  have hP : ScaleInv P := fun c hc y => by
    simp only [P, PiLp.smul_apply, smul_eq_mul, abs_mul, norm_smul, Real.norm_eq_abs,
      abs_of_pos hc]
    constructor
    · intro h; nlinarith
    · intro h; nlinarith
  have hcont : Continuous fun y : E4 => a * ‖y‖ - |y 0| := by fun_prop
  have hm : MeasurableSet {x : S3 | P x} :=
    measurableSet_lt (by fun_prop : Continuous fun x : S3 => |(x : E4) 0|).measurable
      (by fun_prop : Continuous fun x : S3 => a * ‖(x : E4)‖).measurable
  have hmR : MeasurableSet {x : S3 | P (R x)} :=
    measurableSet_lt (by fun_prop : Continuous fun x : S3 => |(R (x : E4)) 0|).measurable
      (by fun_prop : Continuous fun x : S3 => a * ‖R (x : E4)‖).measurable
  have hset : {x : S3 | |⟪n, (x : E4)⟫| < a} = {x : S3 | P (R x)} := by
    ext x
    simp only [P, mem_ofPred_eq, LinearIsometryEquiv.norm_map]
    rw [show ‖(x : E4)‖ = 1 from norm_eq_of_mem_sphere x, mul_one, ← inner_eb, ← hR,
      LinearIsometryEquiv.inner_map_map]
  have hset0 : {x : S3 | P x} = {x : S3 | |(x : E4) 0| < a} := by
    ext x; simp only [P, mem_ofPred_eq]
    rw [show ‖(x : E4)‖ = 1 from norm_eq_of_mem_sphere x, mul_one]
  rw [hset, unifS3_rot R hP hm hmR, hset0, unif_abs_lt_e0 ha0 ha1]

/-! ### Laws on `[0, 1]` given by a distribution function -/

theorem hasDerivAt_Psi {y : ℝ} (h0 : -1 < y) (h1 : y < 1) :
    HasDerivAt Psi (4 / π * √(1 - y ^ 2)) y := by
  have hpos : 0 < 1 - y ^ 2 := by nlinarith
  have hs : HasDerivAt (fun y : ℝ => √(1 - y ^ 2)) (-y / √(1 - y ^ 2)) y := by
    have h := ((hasDerivAt_pow 2 y).const_sub 1).sqrt hpos.ne'
    convert h using 1; field_simp; ring
  have h := (((hasDerivAt_id y).mul hs).add (hasDerivAt_arcsin h0.ne' h1.ne)).const_mul (2 / π)
  have hsq : √(1 - y ^ 2) ^ 2 = 1 - y ^ 2 := Real.sq_sqrt hpos.le
  have hsp : 0 < √(1 - y ^ 2) := Real.sqrt_pos.2 hpos
  convert h using 1
  · funext x; simp only [Psi, Pi.add_apply, Pi.mul_apply, id_eq]
  · simp only [id_eq, one_mul]
    field_simp
    rw [hsq]; ring

theorem Psi_zero : Psi 0 = 0 := by simp [Psi]

theorem Psi_one : Psi 1 = 1 := by
  simp only [Psi, one_pow, sub_self, Real.sqrt_zero, mul_zero, arcsin_one, zero_add]
  field_simp

theorem Psi_continuous : Continuous Psi := by unfold Psi; fun_prop

/-- Two finite measures on ℝ that agree on every `(-∞, b)` and on `ℝ` are equal. -/
theorem ext_of_Iio_real {μ ν : Measure ℝ} [IsFiniteMeasure μ] (h : ∀ b, μ (Iio b) = ν (Iio b))
    (hu : μ univ = ν univ) : μ = ν :=
  ext_of_generate_finite _ (BorelSpace.measurable_eq.trans (borel_eq_generateFrom_Iio ℝ))
    isPiSystem_Iio (by rintro _ ⟨b, rfl⟩; exact h b) hu

/-- The law with density `ρ` on `(0, 1)`. -/
noncomputable def densLaw (ρ : ℝ → ℝ) : Measure ℝ :=
  volume.withDensity ((Ioo (0 : ℝ) 1).indicator fun y => ENNReal.ofReal (ρ y))

theorem densLaw_Iio {ρ F : ℝ → ℝ} (hρ : Continuous ρ) (hρ0 : ∀ y ∈ Ioo (0 : ℝ) 1, 0 ≤ ρ y)
    (hF : ∀ y ∈ Ioo (0 : ℝ) 1, HasDerivAt F (ρ y) y) (hFc : ContinuousOn F (Icc 0 1))
    (hF0 : F 0 = 0) {b : ℝ} (hb0 : 0 < b) (hb1 : b ≤ 1) :
    densLaw ρ (Iio b) = ENNReal.ofReal (F b) := by
  rw [densLaw, withDensity_apply _ measurableSet_Iio, lintegral_indicator measurableSet_Ioo,
    Measure.restrict_restrict measurableSet_Ioo,
    show Ioo (0 : ℝ) 1 ∩ Iio b = Ioo 0 b by
      ext y; simp only [mem_inter_iff, mem_Ioo, mem_Iio]; constructor
      · rintro ⟨⟨h1, -⟩, h3⟩; exact ⟨h1, h3⟩
      · rintro ⟨h1, h2⟩; exact ⟨⟨h1, by linarith⟩, h2⟩]
  rw [← ofReal_integral_eq_lintegral_ofReal
    ((hρ.integrableOn_Icc (a := 0) (b := b)).mono_set Ioo_subset_Icc_self)
    (ae_restrict_of_forall_mem measurableSet_Ioo fun y hy =>
      hρ0 y ⟨hy.1, by linarith [hy.2]⟩)]
  rw [← integral_Ioc_eq_integral_Ioo, ← intervalIntegral.integral_of_le hb0.le,
    integral_eq_sub_of_hasDerivAt_of_le hb0.le (hFc.mono (Icc_subset_Icc le_rfl hb1))
      (fun y hy => hF y ⟨hy.1, by linarith [hy.2]⟩) (hρ.intervalIntegrable _ _), hF0, sub_zero]

/-- `Iio b` is the increasing union of `Iio (b - 1/(n+2))`, so a measure of it is a limit. -/
theorem tendsto_measure_Iio (ν : Measure ℝ) (b : ℝ) :
    Filter.Tendsto (fun n : ℕ => ν (Iio (b - 1 / ((n : ℝ) + 2)))) Filter.atTop
      (nhds (ν (Iio b))) := by
  have hm : Monotone fun n : ℕ => Iio (b - 1 / ((n : ℝ) + 2)) := fun n m hnm => by
    refine Iio_subset_Iio ?_
    have : (n : ℝ) ≤ m := by exact_mod_cast hnm
    gcongr
  have hU : (⋃ n : ℕ, Iio (b - 1 / ((n : ℝ) + 2))) = Iio b := by
    ext y
    simp only [mem_iUnion, mem_Iio]
    constructor
    · rintro ⟨n, hn⟩; have : 0 < 1 / ((n : ℝ) + 2) := by positivity
      linarith
    · intro hy
      obtain ⟨n, hn⟩ := exists_nat_gt (1 / (b - y))
      refine ⟨n, ?_⟩
      have hby : 0 < b - y := by linarith
      have h1 : 1 / ((n : ℝ) + 2) < b - y := by
        rw [div_lt_iff₀ (by positivity)]
        rw [div_lt_iff₀ hby] at hn; nlinarith
      linarith
  have := tendsto_measure_iUnion_atTop (μ := ν) hm
  rwa [hU] at this

/-- **A law from its distribution function.** If `0 ≤ X ≤ 1` and `P(X < b) = F(b)` on `(0, 1)`
with `F(0) = 0`, `F(1) = 1` and `F' = ρ ≥ 0`, the law of `X` has density `ρ` on `(0, 1)`. -/
theorem map_eq_densLaw {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
    {X : Ω → ℝ} (hX : Measurable X) (hX0 : ∀ ω, 0 ≤ X ω) (hX1 : ∀ ω, X ω ≤ 1) {ρ F : ℝ → ℝ}
    (hρ : Continuous ρ)
    (hρ0 : ∀ y ∈ Ioo (0 : ℝ) 1, 0 ≤ ρ y) (hF : ∀ y ∈ Ioo (0 : ℝ) 1, HasDerivAt F (ρ y) y)
    (hFc : ContinuousOn F (Icc 0 1)) (hF0 : F 0 = 0) (hF1 : F 1 = 1)
    (hcdf : ∀ b, 0 < b → b < 1 → μ {ω | X ω < b} = ENNReal.ofReal (F b)) :
    μ.map X = densLaw ρ := by
  have hD := fun {b} => densLaw_Iio hρ hρ0 hF hFc hF0 (b := b)
  have hmap : ∀ b, μ.map X (Iio b) = μ {ω | X ω < b} := fun b => by
    rw [Measure.map_apply hX measurableSet_Iio]; rfl
  have hD1 : densLaw ρ (Iio 1) = 1 := by rw [hD one_pos le_rfl, hF1, ENNReal.ofReal_one]
  have hDu : densLaw ρ univ = 1 := by
    rw [← hD1, densLaw, withDensity_apply _ measurableSet_Iio, withDensity_apply _ MeasurableSet.univ,
      Measure.restrict_univ, ← lintegral_indicator measurableSet_Iio]
    congr 1; funext y
    by_cases hy : y < 1
    · simp [hy]
    · simp only [indicator, mem_Ioo, mem_Iio, hy, and_false, ite_false]
  have hlow : ∀ b, b < 1 → μ.map X (Iio b) = densLaw ρ (Iio b) := by
    intro b hb1
    rw [hmap]
    rcases le_or_gt b 0 with hb | hb
    · have he : {ω | X ω < b} = ∅ := by
        ext ω; simp only [mem_ofPred_eq, mem_empty_iff_false, iff_false, not_lt]
        exact hb.trans (hX0 ω)
      have he' : densLaw ρ (Iio b) = 0 := by
        rw [densLaw, withDensity_apply _ measurableSet_Iio, lintegral_indicator measurableSet_Ioo,
          Measure.restrict_restrict measurableSet_Ioo]
        have : Ioo (0 : ℝ) 1 ∩ Iio b = ∅ := by
          ext y; simp only [mem_inter_iff, mem_Ioo, mem_Iio, mem_empty_iff_false, iff_false]
          intro h; linarith [h.1.1, h.2]
        rw [this, Measure.restrict_empty, lintegral_zero_measure]
      rw [he, he', measure_empty]
    · rw [hcdf b hb hb1, hD hb hb1.le]
  have h1 : μ.map X (Iio 1) = densLaw ρ (Iio 1) := by
    refine tendsto_nhds_unique (tendsto_measure_Iio (μ.map X) 1) ?_
    have : (fun n : ℕ => μ.map X (Iio (1 - 1 / ((n : ℝ) + 2))))
        = fun n : ℕ => densLaw ρ (Iio (1 - 1 / ((n : ℝ) + 2))) :=
      funext fun n => hlow _ (by have : 0 < 1 / ((n : ℝ) + 2) := by positivity
                                 linarith)
    rw [this]; exact tendsto_measure_Iio _ 1
  have hall : ∀ ω, X ω < 2 := fun ω => by linarith [hX1 ω]
  refine ext_of_Iio_real (fun b => ?_) ?_
  · rcases lt_or_ge b 1 with hb1 | hb1
    · exact hlow b hb1
    rcases eq_or_lt_of_le hb1 with rfl | hb1
    · exact h1
    · rw [hmap]
      have hu : {ω | X ω < b} = univ := eq_univ_of_forall fun ω => lt_of_le_of_lt (hX1 ω) hb1
      have h2 : densLaw ρ (Iio b) = 1 := by
        refine le_antisymm ?_ (hD1 ▸ measure_mono (Iio_subset_Iio hb1.le))
        exact hDu ▸ measure_mono (subset_univ _)
      rw [hu, measure_univ, h2]
  · rw [Measure.map_apply hX MeasurableSet.univ, preimage_univ, measure_univ, hDu]

/-- Expectations under a law with density `ρ` on `(0, 1)`. -/
theorem lintegral_densLaw (ρ : ℝ → ℝ) (hρ : Measurable ρ) {g : ℝ → ℝ≥0∞} (hg : Measurable g) :
    ∫⁻ y, g y ∂densLaw ρ = ∫⁻ y in Ioo (0 : ℝ) 1, g y * ENNReal.ofReal (ρ y) := by
  rw [densLaw, lintegral_withDensity_eq_lintegral_mul _
    ((hρ.ennreal_ofReal).indicator measurableSet_Ioo) hg, ← lintegral_indicator measurableSet_Ioo]
  congr 1; funext y
  by_cases hy : y ∈ Ioo (0 : ℝ) 1 <;> simp [indicator, hy, mul_comm]

/-! ### The law of `|⟨n, x⟩|` -/

/-- Fact (i) for every `a > 0`: `P(|⟨n, x⟩| < a) = Ψ(min a 1)`. -/
theorem unif_abs_inner_lt' {n : E4} (hn : ‖n‖ = 1) {a : ℝ} (ha0 : 0 < a) :
    unifS3 {x : S3 | |⟪n, (x : E4)⟫| < a} = ENNReal.ofReal (Psi (min a 1)) := by
  rcases le_or_gt a 1 with ha1 | ha1
  · rw [min_eq_left ha1, unif_abs_inner_lt hn ha0 ha1]
  · rw [min_eq_right ha1.le, Psi_one, ENNReal.ofReal_one]
    have : {x : S3 | |⟪n, (x : E4)⟫| < a} = univ := by
      ext x
      simp only [mem_ofPred_eq, mem_univ, iff_true]
      refine lt_of_le_of_lt ?_ ha1
      have := abs_real_inner_le_norm n (x : E4)
      rwa [hn, norm_eq_of_mem_sphere x, mul_one] at this
    rw [this, measure_univ]

/-- **The law of `|⟨n, x⟩|`** for `x` uniform on S³: density `(4/π) √(1 - t²)` on `(0, 1)`. -/
theorem law_abs_inner {n : E4} (hn : ‖n‖ = 1) :
    unifS3.map (fun x : S3 => |⟪n, (x : E4)⟫|) = densLaw fun y => 4 / π * √(1 - y ^ 2) := by
  have hpi := Real.pi_pos
  refine map_eq_densLaw (by fun_prop) (fun _ => abs_nonneg _) (fun x => ?_) (by fun_prop)
    (fun y _ => by positivity) (fun y hy => hasDerivAt_Psi (by linarith [hy.1]) hy.2)
    Psi_continuous.continuousOn Psi_zero Psi_one (fun b hb0 hb1 => ?_)
  · have := abs_real_inner_le_norm n (x : E4)
    rwa [hn, norm_eq_of_mem_sphere x, mul_one] at this
  · rw [unif_abs_inner_lt hn hb0 hb1.le]

/-! ### Fact (ii): the squared distance to a plane is uniform -/

/-- The inequalities behind the slice of the cone of fact (ii); `k² = b/(1-b)`. -/
theorem ineq_ii {b s w N : ℝ} (hb0 : 0 < b) (hb1 : b < 1) (hs : 0 ≤ s) (hw : 0 ≤ w)
    (hN0 : 0 ≤ N) (hN : N ^ 2 = s ^ 2 + w ^ 2) :
    (0 < N ∧ N < 1 ∧ N ^ 2 - s ^ 2 < b * N ^ 2) ↔
      (w < √(1 - s ^ 2) ∧ w < √(b / (1 - b)) * s) := by
  have hk : √(b / (1 - b)) ^ 2 * (1 - b) = b := by
    rw [Real.sq_sqrt (div_nonneg hb0.le (by linarith))]; field_simp
  have hk0 : 0 ≤ √(b / (1 - b)) := Real.sqrt_nonneg _
  set k := √(b / (1 - b))
  have hb' : 0 < 1 - b := by linarith
  constructor
  · rintro ⟨h0, h1, h2⟩
    refine ⟨?_, ?_⟩
    · rw [Real.lt_sqrt hw]; nlinarith
    · have key : w ^ 2 * (1 - b) < (k * s) ^ 2 * (1 - b) := by nlinarith
      have key' : w ^ 2 < (k * s) ^ 2 := lt_of_mul_lt_mul_right key hb'.le
      exact (pow_lt_pow_iff_left₀ hw (by positivity) two_ne_zero).1 key'
  · rintro ⟨h1, h2⟩
    rw [Real.lt_sqrt hw] at h1
    have hks : 0 < k * s := lt_of_le_of_lt hw h2
    have hspos : 0 < s := by
      rcases eq_or_lt_of_le hs with h | h
      · rw [← h, mul_zero] at hks; exact absurd hks (lt_irrefl 0)
      · exact h
    have hcw : w ^ 2 < (k * s) ^ 2 := pow_lt_pow_left₀ h2 hw two_ne_zero
    refine ⟨?_, ?_, ?_⟩
    · by_contra h; push Not at h
      have : N = 0 := le_antisymm h hN0
      rw [this] at hN; nlinarith
    · nlinarith
    · have : w ^ 2 * (1 - b) < b * s ^ 2 := by nlinarith
      nlinarith

/-- The cone of fact (ii) in coordinates. -/
def coneII (b : ℝ) : Set E4 := {y | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ ‖y‖ ^ 2 - y 0 ^ 2 - y 1 ^ 2 < b * ‖y‖ ^ 2}

theorem coneII_measurable (b : ℝ) : MeasurableSet (coneII b) := by
  have h1 : Continuous fun y : E4 => ‖y‖ := continuous_norm
  have h2 : Continuous fun y : E4 => ‖y‖ ^ 2 - y 0 ^ 2 - y 1 ^ 2 := by fun_prop
  exact (measurableSet_lt measurable_const h1.measurable).inter
    ((measurableSet_lt h1.measurable measurable_const).inter
      (measurableSet_lt h2.measurable ((h1.pow 2).measurable.const_mul b)))

theorem norm_sq_fin2 (u : EuclideanSpace ℝ (Fin 2)) : ‖u‖ ^ 2 = u 0 ^ 2 + u 1 ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity), Fin.sum_univ_two, Real.norm_eq_abs,
    Real.norm_eq_abs, sq_abs, sq_abs]

/-- The radial profile of the cone of fact (ii). -/
noncomputable def GII (b y : ℝ) : ℝ := π * min (√(1 - y ^ 2)) (√(b / (1 - b)) * y) ^ 2

theorem slice_II {b : ℝ} (hb0 : 0 < b) (hb1 : b < 1) (u : EuclideanSpace ℝ (Fin 2)) :
    volume {v : EuclideanSpace ℝ (Fin 2) | joinv 2 2 (u, v) ∈ coneII b}
      = ENNReal.ofReal (GII b ‖u‖) := by
  have hset : {v : EuclideanSpace ℝ (Fin 2) | joinv 2 2 (u, v) ∈ coneII b}
      = ball 0 (min (√(1 - ‖u‖ ^ 2)) (√(b / (1 - b)) * ‖u‖)) := by
    ext v
    have h0 : joinv 2 2 (u, v) 0 = u 0 := joinv_fst 2 2 u v 0
    have h1 : joinv 2 2 (u, v) 1 = u 1 := joinv_fst 2 2 u v 1
    simp only [coneII, mem_ofPred_eq, mem_ball_zero_iff, lt_min_iff]
    rw [h0, h1, show ‖joinv 2 2 (u, v)‖ ^ 2 - u 0 ^ 2 - u 1 ^ 2
        = ‖joinv 2 2 (u, v)‖ ^ 2 - ‖u‖ ^ 2 by rw [norm_sq_fin2]; ring]
    exact ineq_ii hb0 hb1 (norm_nonneg _) (norm_nonneg _) (norm_nonneg _) (joinv_norm_sq 2 2 u v)
  have hm : 0 ≤ min (√(1 - ‖u‖ ^ 2)) (√(b / (1 - b)) * ‖u‖) := by positivity
  rw [hset, EuclideanSpace.volume_ball_fin_two, ← ENNReal.ofReal_pow hm,
    ← ENNReal.ofReal_mul (by positivity), GII, mul_comm]

theorem GII_zero {b y : ℝ} (hy : 1 ≤ y) : GII b y = 0 := by
  have : √(1 - y ^ 2) = 0 := Real.sqrt_eq_zero'.2 (by nlinarith)
  rw [GII, this, min_eq_left (by positivity)]; ring

theorem GII_nonneg (b y : ℝ) : 0 ≤ GII b y := by
  unfold GII; have := Real.pi_pos; positivity

theorem GII_continuous (b : ℝ) : Continuous (GII b) := by
  unfold GII; fun_prop

theorem GII_low {b y : ℝ} (hb0 : 0 < b) (hb1 : b < 1) (hy0 : 0 ≤ y) (hy : y ≤ √(1 - b)) :
    y * GII b y = π * (b / (1 - b)) * y ^ 3 := by
  have hk := Real.sq_sqrt (div_nonneg hb0.le (by linarith : (0 : ℝ) ≤ 1 - b))
  have hy2 : y ^ 2 ≤ 1 - b := by
    have := pow_le_pow_left₀ hy0 hy 2; rwa [Real.sq_sqrt (by linarith)] at this
  have hle : √(b / (1 - b)) * y ≤ √(1 - y ^ 2) := by
    apply Real.le_sqrt_of_sq_le
    rw [mul_pow, hk]
    have : b / (1 - b) * y ^ 2 * (1 - b) ≤ (1 - y ^ 2) * (1 - b) := by
      have e : b / (1 - b) * y ^ 2 * (1 - b) = b * y ^ 2 := by
        field_simp [show (1 - b) ≠ 0 by linarith]
      rw [e]; nlinarith
    exact le_of_mul_le_mul_right this (by linarith)
  rw [GII, min_eq_right hle, mul_pow, hk]; ring

theorem GII_high {b y : ℝ} (hb0 : 0 < b) (hb1 : b < 1) (hy : √(1 - b) ≤ y) (hy1 : y ≤ 1) :
    y * GII b y = π * (y - y ^ 3) := by
  have hy0 : 0 ≤ y := (Real.sqrt_nonneg _).trans hy
  have hy2 : 1 - b ≤ y ^ 2 := by
    have := pow_le_pow_left₀ (Real.sqrt_nonneg _) hy 2; rwa [Real.sq_sqrt (by linarith)] at this
  have hle : √(1 - y ^ 2) ≤ √(b / (1 - b)) * y := by
    rw [show √(b / (1 - b)) * y = √(b / (1 - b) * y ^ 2) by
      rw [Real.sqrt_mul (div_nonneg hb0.le (by linarith)), Real.sqrt_sq hy0]]
    apply Real.sqrt_le_sqrt
    rw [div_mul_eq_mul_div, le_div_iff₀ (by linarith)]; nlinarith
  rw [GII, min_eq_left hle, Real.sq_sqrt (by nlinarith)]; ring

/-- **Volume of the cone of fact (ii).** -/
theorem vol_coneII {b : ℝ} (hb0 : 0 < b) (hb1 : b < 1) :
    volume (coneII b) = ENNReal.ofReal (π ^ 2 / 2 * b) := by
  have hcont : Continuous fun u : EuclideanSpace ℝ (Fin 2) => GII b ‖u‖ :=
    (GII_continuous b).comp continuous_norm
  have hsupp : HasCompactSupport fun u : EuclideanSpace ℝ (Fin 2) => GII b ‖u‖ :=
    HasCompactSupport.intro (isCompact_closedBall 0 1) fun u hu => by
      rw [mem_closedBall_zero_iff, not_le] at hu; exact GII_zero hu.le
  have hint := hcont.integrable_of_hasCompactSupport hsupp (μ := volume)
  have hball : (volume : Measure (EuclideanSpace ℝ (Fin 2))).real (ball 0 1) = π := by
    rw [measureReal_def, EuclideanSpace.volume_ball_fin_two]
    simp [Real.pi_pos.le]
  have hy0 : 0 ≤ √(1 - b) := Real.sqrt_nonneg _
  have hy1 : √(1 - b) ≤ 1 := Real.sqrt_le_one.2 (by linarith)
  have hy2 : √(1 - b) ^ 2 = 1 - b := Real.sq_sqrt (by linarith)
  calc volume (coneII b) = ∫⁻ u, volume {v | joinv 2 2 (u, v) ∈ coneII b} :=
        vol_split 2 2 (coneII_measurable b)
    _ = ∫⁻ u, ENNReal.ofReal (GII b ‖u‖) := lintegral_congr fun u => slice_II hb0 hb1 u
    _ = ENNReal.ofReal (∫ u, GII b ‖u‖) :=
        (ofReal_integral_eq_lintegral_ofReal hint (ae_of_all _ fun u => GII_nonneg b _)).symm
    _ = _ := by
      congr 1
      rw [integral_fun_norm_addHaar volume (GII b)]
      simp only [finrank_euclideanSpace, Fintype.card_fin, hball, smul_eq_mul, nsmul_eq_mul,
        show 2 - 1 = 1 from rfl, pow_one]
      rw [setIntegral_eq_of_subset_of_forall_sdiff_eq_zero measurableSet_Ioi Ioc_subset_Ioi_self
        (fun x hx => by
          simp only [mem_sdiff, mem_Ioi, mem_Ioc, not_and, not_le] at hx
          rw [GII_zero (hx.2 hx.1).le, mul_zero]),
        ← intervalIntegral.integral_of_le zero_le_one,
        ← intervalIntegral.integral_add_adjacent_intervals (b := √(1 - b))
          (f := fun y : ℝ => y * GII b y)
          ((continuous_id.mul (GII_continuous b)).intervalIntegrable _ _)
          ((continuous_id.mul (GII_continuous b)).intervalIntegrable _ _)]
      rw [intervalIntegral.integral_congr (a := 0) (b := √(1 - b))
          (g := fun y => π * (b / (1 - b)) * y ^ 3) (fun y hy => by
          rw [uIcc_of_le hy0] at hy; exact GII_low hb0 hb1 hy.1 hy.2),
        intervalIntegral.integral_congr (a := √(1 - b)) (b := 1)
          (g := fun y => π * (y - y ^ 3)) (fun y hy => by
          rw [uIcc_of_le hy1] at hy; exact GII_high hb0 hb1 hy.1 hy.2)]
      rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul,
        intervalIntegral.integral_sub intervalIntegral.intervalIntegrable_id
          (intervalIntegrable_pow 3), integral_pow, integral_id, integral_pow]
      have h4 : √(1 - b) ^ 4 = (1 - b) ^ 2 := by
        rw [show (4 : ℕ) = 2 * 2 from rfl, pow_mul, hy2]
      have hb' : (1 - b) ≠ 0 := by linarith
      norm_num
      rw [h4, hy2]
      field_simp
      ring

/-- **Fact (ii), coordinate form.** For `x` uniform on S³ and `0 < b < 1`,
`P(x₂² + x₃² < b) = b` (written as `1 - x₀² - x₁²`). -/
theorem unif_dist_lt_e01 {b : ℝ} (hb0 : 0 < b) (hb1 : b < 1) :
    unifS3 {x : S3 | 1 - (x : E4) 0 ^ 2 - (x : E4) 1 ^ 2 < b} = ENNReal.ofReal b := by
  set P : E4 → Prop := fun y => ‖y‖ ^ 2 - y 0 ^ 2 - y 1 ^ 2 < b * ‖y‖ ^ 2
  have hP : ScaleInv P := fun c hc y => by
    simp only [P, PiLp.smul_apply, smul_eq_mul, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
    have hc2 : 0 < c ^ 2 := by positivity
    constructor
    · intro h; nlinarith
    · intro h; nlinarith
  have hset : {x : S3 | 1 - (x : E4) 0 ^ 2 - (x : E4) 1 ^ 2 < b} = {x : S3 | P x} := by
    ext x; simp only [P, mem_ofPred_eq]
    rw [show ‖(x : E4)‖ = 1 from norm_eq_of_mem_sphere x, one_pow, mul_one]
  have hm : MeasurableSet {x : S3 | P x} :=
    measurableSet_lt (by fun_prop : Continuous fun x : S3 =>
        ‖(x : E4)‖ ^ 2 - (x : E4) 0 ^ 2 - (x : E4) 1 ^ 2).measurable
      (by fun_prop : Continuous fun x : S3 => b * ‖(x : E4)‖ ^ 2).measurable
  rw [hset, unifS3_cone hP hm]
  change volume (coneII b) / _ = _
  rw [vol_coneII hb0 hb1, ← ENNReal.ofReal_div_of_pos (by positivity)]
  congr 1
  have := Real.pi_pos
  field_simp

/-- Bessel's inequality for an orthonormal pair. -/
theorem bessel2 {u₁ u₂ x : E4} (h₁ : ‖u₁‖ = 1) (h₂ : ‖u₂‖ = 1) (h₁₂ : ⟪u₁, u₂⟫ = 0) :
    ⟪u₁, x⟫ ^ 2 + ⟪u₂, x⟫ ^ 2 ≤ ‖x‖ ^ 2 := by
  have h := real_inner_self_nonneg (x := x - ⟪u₁, x⟫ • u₁ - ⟪u₂, x⟫ • u₂)
  have e1 : ⟪u₁, u₁⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h₁, one_pow]
  have e2 : ⟪u₂, u₂⟫ = 1 := by rw [real_inner_self_eq_norm_sq, h₂, one_pow]
  have e21 : ⟪u₂, u₁⟫ = 0 := by rw [real_inner_comm, h₁₂]
  simp only [inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right, e1, e2,
    h₁₂, e21] at h
  rw [real_inner_comm u₁ x, real_inner_comm u₂ x] at h
  rw [← real_inner_self_eq_norm_sq]
  nlinarith

/-- The squared distance `1 - ⟨u₁, x⟩² - ⟨u₂, x⟩²` of a point of S³ to the plane of an
orthonormal pair. -/
noncomputable def distSq (u₁ u₂ : E4) (x : S3) : ℝ := 1 - ⟪u₁, (x : E4)⟫ ^ 2 - ⟪u₂, (x : E4)⟫ ^ 2

/-- **Fact (ii) of Proposition 9.** For `x` uniform on S³, an orthonormal pair `u₁, u₂` and
`0 < b < 1`, `P(1 - ⟨u₁, x⟩² - ⟨u₂, x⟩² < b) = b`. -/
theorem unif_distSq_lt {u₁ u₂ : E4} (h₁ : ‖u₁‖ = 1) (h₂ : ‖u₂‖ = 1) (h₁₂ : ⟪u₁, u₂⟫ = 0)
    {b : ℝ} (hb0 : 0 < b) (hb1 : b < 1) :
    unifS3 {x : S3 | distSq u₁ u₂ x < b} = ENNReal.ofReal b := by
  obtain ⟨R, hR1, hR2⟩ := rot_to_e01 h₁ h₂ h₁₂
  set P : E4 → Prop := fun y => ‖y‖ ^ 2 - y 0 ^ 2 - y 1 ^ 2 < b * ‖y‖ ^ 2
  have hP : ScaleInv P := fun c hc y => by
    simp only [P, PiLp.smul_apply, smul_eq_mul, norm_smul, Real.norm_eq_abs, abs_of_pos hc]
    have hc2 : 0 < c ^ 2 := by positivity
    constructor
    · intro h; nlinarith
    · intro h; nlinarith
  have hm : MeasurableSet {x : S3 | P x} :=
    measurableSet_lt (by fun_prop : Continuous fun x : S3 =>
        ‖(x : E4)‖ ^ 2 - (x : E4) 0 ^ 2 - (x : E4) 1 ^ 2).measurable
      (by fun_prop : Continuous fun x : S3 => b * ‖(x : E4)‖ ^ 2).measurable
  have hmR : MeasurableSet {x : S3 | P (R x)} :=
    measurableSet_lt (by fun_prop : Continuous fun x : S3 =>
        ‖R (x : E4)‖ ^ 2 - (R (x : E4)) 0 ^ 2 - (R (x : E4)) 1 ^ 2).measurable
      (by fun_prop : Continuous fun x : S3 => b * ‖R (x : E4)‖ ^ 2).measurable
  have hset : {x : S3 | distSq u₁ u₂ x < b} = {x : S3 | P (R x)} := by
    ext x
    simp only [P, distSq, mem_ofPred_eq, LinearIsometryEquiv.norm_map]
    rw [show ‖(x : E4)‖ = 1 from norm_eq_of_mem_sphere x, one_pow, mul_one, ← inner_eb, ← inner_eb,
      ← hR1, ← hR2, LinearIsometryEquiv.inner_map_map, LinearIsometryEquiv.inner_map_map]
  have hset0 : {x : S3 | P x} = {x : S3 | 1 - (x : E4) 0 ^ 2 - (x : E4) 1 ^ 2 < b} := by
    ext x; simp only [P, mem_ofPred_eq]
    rw [show ‖(x : E4)‖ = 1 from norm_eq_of_mem_sphere x, one_pow, mul_one]
  rw [hset, unifS3_rot R hP hm hmR, hset0, unif_dist_lt_e01 hb0 hb1]

/-- **The law of the squared distance:** uniform on `(0, 1)`. -/
theorem law_distSq {u₁ u₂ : E4} (h₁ : ‖u₁‖ = 1) (h₂ : ‖u₂‖ = 1) (h₁₂ : ⟪u₁, u₂⟫ = 0) :
    unifS3.map (distSq u₁ u₂) = densLaw fun _ => 1 := by
  refine map_eq_densLaw (by unfold distSq; fun_prop) (fun x => ?_) (fun x => ?_) continuous_const
    (fun _ _ => zero_le_one) (fun y _ => hasDerivAt_id y) continuousOn_id rfl rfl
    (fun b hb0 hb1 => ?_)
  · have := bessel2 (x := (x : E4)) h₁ h₂ h₁₂
    rw [norm_eq_of_mem_sphere x, one_pow] at this
    unfold distSq; linarith
  · unfold distSq; nlinarith [sq_nonneg ⟪u₁, (x : E4)⟫, sq_nonneg ⟪u₂, (x : E4)⟫]
  · exact unif_distSq_lt h₁ h₂ h₁₂ hb0 hb1

end Capri
