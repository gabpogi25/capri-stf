/-
Theorem 4.1 (first-order scale family) for the statistic `S = Q_p({D_t})/δ`, and its size and
power in the limit.

* `ostat`: the k-th order statistic (`Tuple.sort`); `ostat_lip`: it is 1-Lipschitz in the sup
  norm; `ostat_smul`: it is positively homogeneous.
* `qlip`, `qsmul`: the same for numpy's interpolated quantile (`interp` of `QuantileInterp`).
* `scale_family_S`: `|S - (r/δ) Z| ≤ C r²/δ` with `Z = Q_p({|Dφ_t(q)(u - v)|})`, uniformly on a
  compact set, for any gauges; `Z` does not involve `r`, `δ` or the gauges.
* `size_power_limit`: if `|S_δ - f Z| ≤ C δ` and `Z` has no atom at `c/f`, then
  `P(S_δ > c) → P(Z > c/f)` as `δ → 0⁺`: size `P(Z > c)` (f = 1) and power `P(Z > c/f)`.
  The power statement needs `Z` to have no atom at `c/f` (for `f = 1`, at `c`).
-/
import CapriFormal.Taylor
import CapriFormal.QuantileInterp
import Mathlib.Probability.CDF

open Filter Topology MeasureTheory ProbabilityTheory Finset

namespace Capri

variable {T : ℕ}

/-- The k-th order statistic (0-based) of `x`. -/
noncomputable def ostat (x : Fin T → ℝ) (k : Fin T) : ℝ := x (Tuple.sort x k)

theorem count_le_ostat (x : Fin T → ℝ) (k : Fin T) :
    k.val + 1 ≤ (univ.filter fun i => x i ≤ ostat x k).card := by
  set σ := Tuple.sort x
  have hmono : Monotone (x ∘ σ) := Tuple.monotone_sort x
  have hsub : (Iic k).image σ ⊆ univ.filter fun i => x i ≤ ostat x k := by
    intro i hi
    obtain ⟨j, hj, rfl⟩ := mem_image.1 hi
    rw [mem_filter]
    exact ⟨mem_univ _, hmono (mem_Iic.1 hj)⟩
  have := card_le_card hsub
  rw [card_image_of_injective _ σ.injective, Fin.card_Iic] at this
  exact this

theorem ostat_le_of_count (y : Fin T → ℝ) (k : Fin T) (c : ℝ)
    (h : k.val + 1 ≤ (univ.filter fun i => y i ≤ c).card) : ostat y k ≤ c := by
  by_contra hc
  push Not at hc
  have := card_le_of_interp_gt y c (k : ℝ) k k (by simp) le_rfl
    (by simpa [interp, ostat] using hc)
  omega

/-- Order statistics are 1-Lipschitz in the sup norm. -/
theorem ostat_lip (x y : Fin T → ℝ) {ε : ℝ} (h : ∀ i, |x i - y i| ≤ ε) (k : Fin T) :
    |ostat x k - ostat y k| ≤ ε := by
  have one : ∀ a b : Fin T → ℝ, (∀ i, |a i - b i| ≤ ε) → ostat b k ≤ ostat a k + ε := by
    intro a b hab
    refine ostat_le_of_count b k _ (le_trans (count_le_ostat a k) (card_le_card ?_))
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    have := (abs_le.1 (hab i)).1
    linarith
  have h1 := one x y h
  have h2 := one y x (fun i => by rw [abs_sub_comm]; exact h i)
  rw [abs_le]; constructor <;> linarith

/-- Order statistics are positively homogeneous. -/
theorem ostat_smul (x : Fin T → ℝ) {r : ℝ} (hr : 0 ≤ r) (k : Fin T) :
    ostat (fun i => r * x i) k = r * ostat x k := by
  rcases hr.lt_or_eq with hr | hr
  · apply le_antisymm
    · refine ostat_le_of_count _ k _ (le_trans (count_le_ostat x k) (card_le_card ?_))
      intro i hi
      simp only [mem_filter, mem_univ, true_and] at hi ⊢
      exact mul_le_mul_of_nonneg_left hi hr.le
    · have : ostat x k ≤ ostat (fun i => r * x i) k / r := by
        refine ostat_le_of_count x k _ (le_trans (count_le_ostat (fun i => r * x i) k) (card_le_card ?_))
        intro i hi
        simp only [mem_filter, mem_univ, true_and] at hi ⊢
        rw [le_div_iff₀ hr, mul_comm]; exact hi
      rwa [le_div_iff₀ hr, mul_comm] at this
  · subst hr; simp [ostat]

/-- numpy's interpolated quantile, `Q = (1 - θ) x_(lo) + θ x_(hi)`, `θ = h - lo`. -/
noncomputable def qp (x : Fin T → ℝ) (h : ℝ) (lo hi : Fin T) : ℝ :=
  interp (x ∘ Tuple.sort x) h lo hi

theorem qp_eq (x : Fin T → ℝ) (h : ℝ) (lo hi : Fin T) :
    qp x h lo hi = (1 - (h - lo)) * ostat x lo + (h - lo) * ostat x hi := by
  simp only [qp, interp, ostat, Function.comp]; ring

/-- The interpolated quantile is 1-Lipschitz in the sup norm. -/
theorem qlip (x y : Fin T → ℝ) {ε : ℝ} (hxy : ∀ i, |x i - y i| ≤ ε) (h : ℝ) (lo hi : Fin T)
    (h0 : 0 ≤ h - lo) (h1 : h - lo ≤ 1) : |qp x h lo hi - qp y h lo hi| ≤ ε := by
  rw [qp_eq, qp_eq]
  have a := ostat_lip x y hxy lo
  have b := ostat_lip x y hxy hi
  have e : (1 - (h - lo)) * ostat x lo + (h - lo) * ostat x hi
      - ((1 - (h - lo)) * ostat y lo + (h - lo) * ostat y hi)
      = (1 - (h - lo)) * (ostat x lo - ostat y lo) + (h - lo) * (ostat x hi - ostat y hi) := by
    ring
  rw [e]
  calc _ ≤ |(1 - (h - lo)) * (ostat x lo - ostat y lo)| + |(h - lo) * (ostat x hi - ostat y hi)| :=
        abs_add_le _ _
    _ = (1 - (h - lo)) * |ostat x lo - ostat y lo| + (h - lo) * |ostat x hi - ostat y hi| := by
        rw [abs_mul, abs_mul, abs_of_nonneg (by linarith), abs_of_nonneg h0]
    _ ≤ (1 - (h - lo)) * ε + (h - lo) * ε := by
        gcongr
    _ = ε := by ring

theorem qsmul (x : Fin T → ℝ) {r : ℝ} (hr : 0 ≤ r) (h : ℝ) (lo hi : Fin T) :
    qp (fun i => r * x i) h lo hi = r * qp x h lo hi := by
  rw [qp_eq, qp_eq, ostat_smul x hr, ostat_smul x hr]; ring

variable {n : ℕ}

/-- **Theorem 4.1 for the statistic.** For a family of `T` pencils whose brackets are nonzero on
a compact set `K`, noise `‖u‖, ‖v‖ ≤ B`, any gauges and `δ > 0`:
`|S - (r/δ) Z| ≤ C r²/δ` with `S = Q_p({D_t})/δ` and `Z = Q_p({|Dφ_t(q)(u - v)|})`. -/
theorem scale_family_S (ts : Fin T → Fin 6 → Fin n) {K : Set (Config n)} (hK : IsCompact K)
    (hKU : ∀ i, K ⊆ Ut (ts i)) {B : ℝ} (hB : 0 ≤ B) (h : ℝ) (lo hi : Fin T)
    (h0 : 0 ≤ h - lo) (h1 : h - lo ≤ 1) :
    ∃ C r₀ : ℝ, 0 ≤ C ∧ 0 < r₀ ∧ ∀ q ∈ K, ∀ u v : Config n, ‖u‖ ≤ B → ‖v‖ ≤ B →
      ∀ r, 0 ≤ r → r ≤ r₀ → ∀ (A₁ A₂ : Matrix (Fin 4) (Fin 4) ℝ) (l₁ l₂ : Fin n → ℝ),
      A₁.det ≠ 0 → A₂.det ≠ 0 → (∀ k, l₁ k ≠ 0) → (∀ k, l₂ k ≠ 0) → ∀ δ : ℝ, 0 < δ →
      |qp (fun i => |Real.log (crAt (ts i) (act A₁ l₁ (q + r • u)))
            - Real.log (crAt (ts i) (act A₂ l₂ (q + r • v)))|) h lo hi / δ
        - r / δ * qp (fun i => |fderiv ℝ (fun Q => Real.log (crAt (ts i) Q)) q (u - v)|) h lo hi|
        ≤ C * r ^ 2 / δ := by
  classical
  choose C r₀ hC hr₀ hbound using fun i => scale_family (ts i) hK (hKU i) hB
  rcases Nat.eq_zero_or_pos T with hT | hT
  · subst hT; exact lo.elim0
  have hne : (univ : Finset (Fin T)).Nonempty := ⟨lo, mem_univ _⟩
  refine ⟨univ.sup' hne C, univ.inf' hne r₀, le_trans (hC lo) (le_sup' C (mem_univ lo)),
    (lt_inf'_iff _).2 fun i _ => hr₀ i, ?_⟩
  intro q hq u v hu hv r hr0 hr A₁ A₂ l₁ l₂ hA₁ hA₂ hl₁ hl₂ δ hδ
  set D := fun i => |Real.log (crAt (ts i) (act A₁ l₁ (q + r • u)))
    - Real.log (crAt (ts i) (act A₂ l₂ (q + r • v)))|
  set g := fun i => |fderiv ℝ (fun Q => Real.log (crAt (ts i) Q)) q (u - v)|
  have hpt : ∀ i, |D i - r * g i| ≤ univ.sup' hne C * r ^ 2 := by
    intro i
    refine le_trans (hbound i q hq u v hu hv r hr0 (le_trans hr (inf'_le _ (mem_univ i)))
      A₁ A₂ l₁ l₂ hA₁ hA₂ hl₁ hl₂) ?_
    exact mul_le_mul_of_nonneg_right (le_sup' C (mem_univ i)) (sq_nonneg r)
  have hq := qlip D (fun i => r * g i) hpt h lo hi h0 h1
  rw [qsmul g hr0] at hq
  have e : qp D h lo hi / δ - r / δ * qp g h lo hi = (qp D h lo hi - r * qp g h lo hi) / δ := by
    field_simp
  rw [e, abs_div, abs_of_pos hδ]
  exact div_le_div_of_nonneg_right hq hδ.le

/-- **Size and power in the limit.** If `|S_δ - f Z| ≤ C δ` for every `δ > 0` and `Z` has no
atom at `c/f`, then `P(S_δ > c) → P(Z > c/f)` as `δ → 0⁺`. -/
theorem size_power_limit {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    (Z : Ω → ℝ) (hZ : Measurable Z) (S : ℝ → Ω → ℝ) {C f c : ℝ} (hf : 0 < f)
    (hS : ∀ δ, 0 < δ → ∀ ω, |S δ ω - f * Z ω| ≤ C * δ) (hatom : P {ω | Z ω = c / f} = 0) :
    Tendsto (fun δ => P.real {ω | c < S δ ω}) (𝓝[>] 0) (𝓝 (P.real {ω | c / f < Z ω})) := by
  have hprob : IsProbabilityMeasure (P.map Z) :=
    (Measure.isProbabilityMeasure_map_iff hZ.aemeasurable).2 inferInstance
  set F := cdf (P.map Z)
  have hFval : ∀ x, P.real {ω | x < Z ω} = 1 - F x := by
    intro x
    rw [cdf_eq_real, measureReal_def (μ := P.map Z), Measure.map_apply hZ measurableSet_Iic]
    have h := measureReal_compl (μ := P) (s := Z ⁻¹' Set.Iic x) (hZ measurableSet_Iic)
    rw [probReal_univ] at h
    have e : (Z ⁻¹' Set.Iic x)ᶜ = {ω | x < Z ω} := by ext ω; simp
    rw [e] at h; rw [h, measureReal_def]
  -- F is continuous at c/f
  have hcont : ContinuousAt F (c / f) := by
    have hmono : Monotone F := F.mono
    have hsing : F.measure {c / f} = 0 := by
      rw [measure_cdf, Measure.map_apply hZ (measurableSet_singleton _)]
      have e : Z ⁻¹' {c / f} = {ω | Z ω = c / f} := rfl
      rw [e, hatom]
    rw [StieltjesFunction.measure_singleton, ENNReal.ofReal_eq_zero] at hsing
    have hleft : Function.leftLim F (c / f) = F (c / f) :=
      le_antisymm (hmono.leftLim_le le_rfl) (by linarith)
    have hl : ContinuousWithinAt F (Set.Iio (c / f)) (c / f) :=
      (hmono.continuousWithinAt_Iio_iff_leftLim_eq).2 hleft
    rw [continuousAt_iff_continuous_left_right]
    exact ⟨continuousWithinAt_Iio_iff_Iic.1 hl, F.right_continuous _⟩
  have hlo : Tendsto (fun δ => 1 - F ((c + C * δ) / f)) (𝓝[>] 0) (𝓝 (1 - F (c / f))) := by
    have : Tendsto (fun δ : ℝ => (c + C * δ) / f) (𝓝[>] 0) (𝓝 (c / f)) := by
      have hc : Continuous fun δ : ℝ => (c + C * δ) / f := by fun_prop
      simpa using (hc.tendsto 0).mono_left nhdsWithin_le_nhds
    exact (hcont.tendsto.comp this).const_sub 1
  have hhi : Tendsto (fun δ => 1 - F ((c - C * δ) / f)) (𝓝[>] 0) (𝓝 (1 - F (c / f))) := by
    have : Tendsto (fun δ : ℝ => (c - C * δ) / f) (𝓝[>] 0) (𝓝 (c / f)) := by
      have hc : Continuous fun δ : ℝ => (c - C * δ) / f := by fun_prop
      simpa using (hc.tendsto 0).mono_left nhdsWithin_le_nhds
    exact (hcont.tendsto.comp this).const_sub 1
  rw [hFval]
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' hlo hhi ?_ ?_
  · filter_upwards [self_mem_nhdsWithin] with δ (hδ : 0 < δ)
    rw [← hFval]
    refine measureReal_mono fun ω (hω : (c + C * δ) / f < Z ω) => ?_
    show c < S δ ω
    have := (abs_le.1 (hS δ hδ ω)).1
    rw [div_lt_iff₀ hf] at hω
    linarith
  · filter_upwards [self_mem_nhdsWithin] with δ (hδ : 0 < δ)
    rw [← hFval]
    refine measureReal_mono fun ω (hω : c < S δ ω) => ?_
    show (c - C * δ) / f < Z ω
    have := (abs_le.1 (hS δ hδ ω)).2
    rw [div_lt_iff₀ hf]
    linarith

end Capri
