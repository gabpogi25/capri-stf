/-
Proposition 5 (MATHEMATICS.md Proposition 4.4), assembled: with `ũ, ṽ` independent standard Gaussian
in the 4n coordinates, the standardised discrepancy `z_t` converges in distribution to
`|N(0,1)|/(√2 μ)` as `r → 0`, uniformly in `q` over a compact set of unit configurations on
which the brackets of `t` are nonzero and `‖g_t‖ ≥ γ > 0`.
-/
import CapriFormal.Pivotal
import CapriFormal.LimitLaw
import Mathlib.MeasureTheory.Measure.TightNormed

open MeasureTheory ProbabilityTheory Filter Topology
open scoped RealInnerProductSpace

namespace Capri

variable {n : ℕ}

/-- The coordinate space ℝ^{4n} with its Euclidean structure. -/
abbrev Coord (n : ℕ) := EuclideanSpace ℝ (Fin n × Fin 4)

/-- Coordinates to configuration. -/
def toCfg (x : Coord n) : Config n := fun k j => x (k, j)

theorem toCfg_sub (x y : Coord n) : toCfg (x - y) = toCfg x - toCfg y := by
  funext k j; simp [toCfg]

theorem toCfg_norm_le (x : Coord n) : ‖toCfg x‖ ≤ ‖x‖ := by
  refine (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2 fun k => ?_
  refine (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2 fun j => ?_
  have := PiLp.norm_apply_le x (k, j)
  simpa [toCfg] using this

theorem toCfg_eq_sum (x : Coord n) : toCfg x = ∑ kj : Fin n × Fin 4, x kj • ec kj.1 kj.2 := by
  funext k j
  simp only [toCfg, Finset.sum_apply, Pi.smul_apply, ec, ev, smul_eq_mul]
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_eq_single k]
  · rw [Finset.sum_eq_single j]
    · simp
    · intro b _ hb; simp [Pi.single_apply, hb.symm]
    · simp
  · intro b _ hb; simp [Pi.single_apply, Ne.symm hb]
  · simp

/-- A derivative applied to coordinates is the inner product with the gradient. -/
theorem apply_toCfg (L : Config n →L[ℝ] ℝ) (x : Coord n) :
    L (toCfg x) = ⟪(WithLp.toLp 2 (fun kj : Fin n × Fin 4 => L (ec kj.1 kj.2)) : Coord n), x⟫ := by
  rw [toCfg_eq_sum, map_sum, PiLp.inner_apply]
  refine Finset.sum_congr rfl fun kj _ => ?_
  simp [map_smul, mul_comm]

theorem Lstat_eq (t : Fin 6 → Fin n) (μ : ℝ) (q : Config n) (u v : Coord n) :
    Lstat t μ q (toCfg u) (toCfg v) =
      |⟪grad (phi t) q, u - v⟫| / (2 * μ * ‖grad (phi t) q‖) := by
  simp only [Lstat]
  rw [← toCfg_sub, apply_toCfg]
  rfl

/-- **Proposition 5, general form.** Any statistic `z` that is within `C_β r` of `L_t` whenever
the noise is bounded by `β` (and `r ≤ r₀(β)`) converges in distribution to `|N(0,1)|/(√2 μ)`,
uniformly in `q ∈ K`. -/
theorem pivotal_limit_of (t : Fin 6 → Fin n) {K : Set (Config n)}
    {γ μ : ℝ} (hgrad : ∀ q ∈ K, γ ≤ ‖grad (phi t) q‖) (hγ : 0 < γ) (hμ : 0 < μ)
    (z : Config n → ℝ → Coord n × Coord n → ℝ)
    (hz : ∀ β, 0 ≤ β → ∃ C r₀ : ℝ, 0 < r₀ ∧ ∀ q ∈ K, ∀ ω : Coord n × Coord n,
      ‖toCfg ω.1‖ ≤ β → ‖toCfg ω.2‖ ≤ β → ∀ r, 0 < r → r ≤ r₀ →
        |z q r ω - Lstat t μ q (toCfg ω.1) (toCfg ω.2)| ≤ C * r) :
    ∀ ε > 0, ∃ r₁ > 0, ∀ r, 0 < r → r ≤ r₁ → ∀ q ∈ K, ∀ x : ℝ,
      |((stdGaussian (Coord n)).prod (stdGaussian (Coord n))).real {ω | z q r ω ≤ x}
        - (gaussianReal 0 1).real {z | |z| / (Real.sqrt 2 * μ) ≤ x}| ≤ ε := by
  set G := stdGaussian (Coord n)
  set P := G.prod G
  set F : ℝ → ℝ := fun x => (gaussianReal 0 1).real {z | |z| / (Real.sqrt 2 * μ) ≤ x}
  -- the good events
  set E : ℝ → Set (Coord n × Coord n) :=
    fun β => {ω | ‖toCfg ω.1‖ ≤ β ∧ ‖toCfg ω.2‖ ≤ β}
  have htail : Tendsto (fun β : ℝ => G.real (Metric.closedBall 0 β)ᶜ) atTop (𝓝 0) := by
    have h := tendsto_measure_compl_closedBall_of_isTightMeasureSet
      (isTightMeasureSet_singleton (μ := G)) (0 : Coord n)
    simp only [Set.mem_singleton_iff, iSup_iSup_eq_left] at h
    have := (ENNReal.tendsto_toReal ENNReal.zero_ne_top).comp h
    simpa [Function.comp_def, measureReal_def] using this
  have hE : Tendsto (fun β => P.real (E β)ᶜ) atTop (𝓝 0) := by
    have hle : ∀ β, P.real (E β)ᶜ ≤ 2 * G.real (Metric.closedBall 0 β)ᶜ := by
      intro β
      have hsub : (E β)ᶜ ⊆ Prod.fst ⁻¹' (Metric.closedBall 0 β)ᶜ ∪
          Prod.snd ⁻¹' (Metric.closedBall 0 β)ᶜ := by
        intro ω hω
        simp only [E, Set.mem_compl_iff, Set.mem_setOf_eq, not_and_or, not_le] at hω
        simp only [Set.mem_union, Set.mem_preimage, Set.mem_compl_iff, Metric.mem_closedBall,
          dist_zero_right, not_le]
        rcases hω with h | h
        · exact Or.inl (lt_of_lt_of_le h (toCfg_norm_le _))
        · exact Or.inr (lt_of_lt_of_le h (toCfg_norm_le _))
      have h1 : P.real (Prod.fst ⁻¹' (Metric.closedBall 0 β)ᶜ) = G.real (Metric.closedBall 0 β)ᶜ := by
        rw [measureReal_def, measureReal_def, ← Measure.map_apply measurable_fst
          Metric.isClosed_closedBall.isOpen_compl.measurableSet]
        simp [P]
      have h2 : P.real (Prod.snd ⁻¹' (Metric.closedBall 0 β)ᶜ) = G.real (Metric.closedBall 0 β)ᶜ := by
        rw [measureReal_def, measureReal_def, ← Measure.map_apply measurable_snd
          Metric.isClosed_closedBall.isOpen_compl.measurableSet]
        simp [P]
      calc P.real (E β)ᶜ ≤ P.real (Prod.fst ⁻¹' (Metric.closedBall 0 β)ᶜ ∪
            Prod.snd ⁻¹' (Metric.closedBall 0 β)ᶜ) := measureReal_mono hsub
        _ ≤ _ := measureReal_union_le _ _
        _ = 2 * G.real (Metric.closedBall 0 β)ᶜ := by rw [h1, h2]; ring
    have h2 : Tendsto (fun β => 2 * G.real (Metric.closedBall 0 β)ᶜ) atTop (𝓝 0) := by
      simpa using htail.const_mul 2
    exact tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds h2
      (fun β => measureReal_nonneg) hle
  -- the Taylor step on the good events
  have hloc : ∀ β, ∃ C r₀ : ℝ, 0 < r₀ ∧ ∀ r, 0 < r → r ≤ r₀ → ∀ q : K, ∀ ω ∈ E β,
      |z q r ω - Lstat t μ q (toCfg ω.1) (toCfg ω.2)| ≤ C * r := by
    intro β
    obtain ⟨C, r₀, hr₀, h⟩ := hz (max β 0) (le_max_right β 0)
    exact ⟨C, r₀, hr₀, fun r hr hrr q ω hω => h q q.2 ω (le_trans hω.1 (le_max_left _ _))
      (le_trans hω.2 (le_max_left _ _)) r hr hrr⟩
  -- the law of L_t: |N(0,1)|/(√2 μ) for every q
  have hF : ∀ (q : K) (x : ℝ),
      P.real {ω | Lstat t μ q (toCfg ω.1) (toCfg ω.2) ≤ x} = F x := by
    intro q x
    set g := grad (phi t) (q : Config n)
    have hg : g ≠ 0 := by
      intro h; have := hgrad q q.2; rw [show ‖grad (phi t) (q : Config n)‖ = ‖g‖ from rfl, h,
        norm_zero] at this; linarith
    have hind : IndepFun Prod.fst Prod.snd P := by
      rw [indepFun_iff_map_prod_eq_prod_map_map measurable_fst.aemeasurable
        measurable_snd.aemeasurable]
      simp [P]
    have hlaw := sub_stdGaussian (P := P) measurable_fst measurable_snd (by simp [P, G])
      (by simp [P, G]) hind
    have hll := limit_law hg hμ
    set h : Coord n → ℝ := fun w => |⟪g, w⟫| / (2 * μ * ‖g‖)
    have hset : {ω : Coord n × Coord n | Lstat t μ q (toCfg ω.1) (toCfg ω.2) ≤ x}
        = (fun ω : Coord n × Coord n => ω.1 - ω.2) ⁻¹' (h ⁻¹' Set.Iic x) := by
      ext ω; simp [Lstat_eq, h, g]
    have hmh : Measurable h := by fun_prop
    rw [hset, measureReal_def, ← Measure.map_apply (by fun_prop) (hmh measurableSet_Iic), hlaw,
      Measure.map_apply (by fun_prop) (hmh measurableSet_Iic), ← Set.preimage_comp,
      ← Measure.map_apply (by fun_prop) measurableSet_Iic]
    have e : (h ∘ fun x => Real.sqrt 2 • x) = fun x => |⟪g, Real.sqrt 2 • x⟫| / (2 * μ * ‖g‖) := rfl
    rw [e, hll, Measure.map_apply (by fun_prop) measurableSet_Iic]
    rfl
  -- the modulus of the limit law
  have hmod : ∀ ε > 0, ∃ η > 0, ∀ x, F (x + η) - F (x - η) ≤ ε := by
    intro ε hε
    have hs : 0 < Real.sqrt 2 * μ := mul_pos (Real.sqrt_pos.2 two_pos) hμ
    refine ⟨ε / (4 * (Real.sqrt 2 * μ)), by positivity, fun x => ?_⟩
    refine le_trans (half_normal_modulus hμ (by positivity) x) (le_of_eq ?_)
    field_simp
  intro ε hε
  obtain ⟨r₁, hr₁, h⟩ := limit_theorem_uniform P
    (fun (q : K) r ω => z q r ω)
    (fun (q : K) ω => Lstat t μ q (toCfg ω.1) (toCfg ω.2)) E hE hloc F hF hmod ε hε
  exact ⟨r₁, hr₁, fun r hr hrr q hq x => h r hr hrr ⟨q, hq⟩ x⟩

/-- **Proposition 5.** With `ũ, ṽ` independent standard Gaussian in the 4n coordinates and
`m₀ = μ r`, the standardised discrepancy `z_t` converges in distribution to `|N(0,1)|/(√2 μ)` as
`r → 0`, uniformly over a compact set of unit configurations on which the brackets of `t` are
nonzero and `‖g_t‖ ≥ γ > 0`. -/
theorem pivotal_limit (t : Fin 6 → Fin n) {K : Set (Config n)} (hK : IsCompact K)
    (hKU : K ⊆ Ut t) (hKunit : ∀ q ∈ K, ∀ k, ‖WithLp.toLp 2 (q k)‖ = 1)
    {γ μ : ℝ} (hγ : 0 < γ) (hgrad : ∀ q ∈ K, γ ≤ ‖grad (phi t) q‖) (hμ : 0 < μ) :
    ∀ ε > 0, ∃ r₁ > 0, ∀ r, 0 < r → r ≤ r₁ → ∀ q ∈ K, ∀ x : ℝ,
      |((stdGaussian (Coord n)).prod (stdGaussian (Coord n))).real
          {ω | zstat t μ r q (toCfg ω.1) (toCfg ω.2) ≤ x}
        - (gaussianReal 0 1).real {z | |z| / (Real.sqrt 2 * μ) ≤ x}| ≤ ε := by
  refine pivotal_limit_of t hgrad hγ hμ (fun q r ω => zstat t μ r q (toCfg ω.1) (toCfg ω.2)) ?_
  intro β hβ
  obtain ⟨C, r₀, -, hr₀, h⟩ := pivotal_taylor t hK hKU hKunit hγ hgrad hμ hβ
  exact ⟨C, r₀, hr₀, fun q hq ω h1 h2 r hr hrr => h q hq _ _ h1 h2 r hr hrr⟩

end Capri
