/-
The probabilistic core of Proposition 5 (MATHEMATICS.md Proposition 4.4): the limit law of the
standardised discrepancies, and the uniform convergence argument.

* `sub_stdGaussian`: if `U, V` are independent standard Gaussian vectors, `U - V` has the law of
  `√2 X` with `X` standard Gaussian (that is, `w = u~ - v~ ~ N(0, 2I)`).
* `stdGaussian_map_inner_div`: for `g ≠ 0`, `⟪g, X⟫ / ‖g‖ ~ N(0, 1)`.
* `limit_law`: `|⟪g, w⟫| / (2 μ ‖g‖)` has the law of `|N(0,1)| / (√2 μ)`, whatever `g ≠ 0`:
  the limit law does not depend on the configuration or the pencil.
* `cdf_coupling`: `|P(Z ≤ x) - P(L ≤ x)| ≤ P(|Z - L| > η) + P(L ≤ x + η) - P(L ≤ x - η)`.
* `uniform_in_prob`: on events `E_β` whose complements have probability → 0, a bound
  `|z - L| ≤ C_β r` for `r ≤ r₀(β)`, uniform in the configuration, gives
  `sup_q P(|z - L| > η) → 0`.
* `half_normal_modulus`: the distribution function of `|N(0,1)| / (√2 μ)` moves by at most
  `4 √2 μ η` over an interval of half-width `η`.
* `limit_theorem_uniform`: Proposition 5's conclusion, `sup_q sup_x |P(z ≤ x) - F(x)| → 0`.

Not formalized: the Taylor step, that is the bound `|z - L| ≤ C_β r` on `E_β` (it is the
hypothesis `hloc`); it rests on the first-order expansion of Theorem 4 and the smoothness of
the gradient.
-/
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Independence.CharacteristicFunction

open MeasureTheory ProbabilityTheory Filter Topology Complex
open scoped RealInnerProductSpace

namespace Capri

section Gaussian

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [MeasurableSpace E] [BorelSpace E]

/-- The difference of two independent standard Gaussian vectors is `√2` times a standard
Gaussian vector. -/
theorem sub_stdGaussian {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
    {U V : Ω → E} (mU : Measurable U) (mV : Measurable V) (hU : P.map U = stdGaussian E)
    (hV : P.map V = stdGaussian E) (hUV : IndepFun U V P) :
    P.map (fun ω => U ω - V ω) = (stdGaussian E).map (fun x => Real.sqrt 2 • x) := by
  have hnV : P.map (-V) = (stdGaussian E).map (fun x => (-1 : ℝ) • x) := by
    rw [← hV, Measure.map_map (by fun_prop) mV]
    congr 1; funext ω; simp
  have hind : IndepFun U (-V) P := hUV.comp measurable_id measurable_neg
  apply Measure.ext_of_charFun
  funext t
  have hsub : (fun ω => U ω - V ω) = fun ω => U ω + (-V) ω := by
    funext ω; simp [sub_eq_add_neg]
  rw [hsub, hind.charFun_map_fun_add_eq_mul mU.aemeasurable mV.neg.aemeasurable, Pi.mul_apply,
    hU, hnV, charFun_map_smul, charFun_map_smul, charFun_stdGaussian, charFun_stdGaussian,
    charFun_stdGaussian, ← Complex.exp_add]
  congr 1
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_neg, abs_one, one_mul,
    abs_of_pos (Real.sqrt_pos.2 two_pos)]
  have h2 : ((Real.sqrt 2 : ℝ) : ℂ) ^ 2 = 2 := by
    rw [← Complex.ofReal_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]; norm_num
  push_cast
  rw [mul_pow, h2]
  ring

/-- For `g ≠ 0`, the normalised projection `⟪g, X⟫ / ‖g‖` of a standard Gaussian vector is a
standard normal variable. -/
theorem stdGaussian_map_inner_div {g : E} (hg : g ≠ 0) :
    (stdGaussian E).map (fun x => ⟪g, x⟫ / ‖g‖) = gaussianReal 0 1 := by
  set L : StrongDual ℝ E := ‖g‖⁻¹ • innerSL ℝ g with hLdef
  have hL : (fun x => ⟪g, x⟫ / ‖g‖) = ⇑L := by
    funext x; simp [L, div_eq_inv_mul]
  have hnorm : ‖L‖ = 1 := by
    rw [hLdef, norm_smul, innerSL_apply_norm, norm_inv, norm_norm,
      inv_mul_cancel₀ (norm_ne_zero_iff.2 hg)]
  rw [hL, IsGaussian.map_eq_gaussianReal, integral_strongDual_stdGaussian,
    variance_dual_stdGaussian, hnorm]
  simp

/-- **Proposition 5, the limit law.** With `w = √2 X` (law `N(0, 2I)`), the variable
`|⟪g, w⟫| / (2 μ ‖g‖)` has the law of `|N(0,1)| / (√2 μ)` for every `g ≠ 0`. -/
theorem limit_law {g : E} (hg : g ≠ 0) {μ : ℝ} (hμ : 0 < μ) :
    (stdGaussian E).map (fun x => |⟪g, Real.sqrt 2 • x⟫| / (2 * μ * ‖g‖))
      = (gaussianReal 0 1).map (fun z => |z| / (Real.sqrt 2 * μ)) := by
  have hn : 0 < ‖g‖ := norm_pos_iff.2 hg
  have hs : 0 < Real.sqrt 2 := Real.sqrt_pos.2 two_pos
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := Real.mul_self_sqrt (by norm_num)
  have h : (fun x => |⟪g, Real.sqrt 2 • x⟫| / (2 * μ * ‖g‖))
      = (fun z => |z| / (Real.sqrt 2 * μ)) ∘ (fun x => ⟪g, x⟫ / ‖g‖) := by
    funext x
    simp only [Function.comp, inner_smul_right, abs_mul, abs_div, abs_norm, abs_of_pos hs]
    field_simp
    rw [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    ring
  rw [h, ← Measure.map_map (by fun_prop) (by fun_prop), stdGaussian_map_inner_div hg]

end Gaussian

section Coupling

variable {Ω : Type*} [MeasurableSpace Ω]

/-- **Coupling.** If `Z` is close to `L` outside a small event, their distribution functions
are close: `|P(Z ≤ x) - P(L ≤ x)| ≤ P(|Z - L| > η) + (P(L ≤ x + η) - P(L ≤ x - η))`. -/
theorem cdf_coupling (P : Measure Ω) [IsFiniteMeasure P] (Z L : Ω → ℝ) {η : ℝ} (hη : 0 ≤ η)
    (x : ℝ) :
    |P.real {ω | Z ω ≤ x} - P.real {ω | L ω ≤ x}|
      ≤ P.real {ω | η < |Z ω - L ω|}
        + (P.real {ω | L ω ≤ x + η} - P.real {ω | L ω ≤ x - η}) := by
  have h1 : P.real {ω | Z ω ≤ x} ≤ P.real {ω | L ω ≤ x + η} + P.real {ω | η < |Z ω - L ω|} := by
    refine le_trans (measureReal_mono ?_) (measureReal_union_le _ _)
    intro ω hω
    by_cases h : η < |Z ω - L ω|
    · exact Or.inr h
    · left
      simp only [Set.mem_ofPred_eq] at hω ⊢
      have := (abs_le.1 (not_lt.1 h)).1
      linarith
  have h2 : P.real {ω | L ω ≤ x - η} ≤ P.real {ω | Z ω ≤ x} + P.real {ω | η < |Z ω - L ω|} := by
    refine le_trans (measureReal_mono ?_) (measureReal_union_le _ _)
    intro ω hω
    by_cases h : η < |Z ω - L ω|
    · exact Or.inr h
    · left
      simp only [Set.mem_ofPred_eq] at hω ⊢
      have := (abs_le.1 (not_lt.1 h)).2
      linarith
  have h3 : P.real {ω | L ω ≤ x} ≤ P.real {ω | L ω ≤ x + η} :=
    measureReal_mono fun ω (hω : L ω ≤ x) => show L ω ≤ x + η by linarith
  have h4 : P.real {ω | L ω ≤ x - η} ≤ P.real {ω | L ω ≤ x} :=
    measureReal_mono fun ω (hω : L ω ≤ x - η) => show L ω ≤ x by linarith
  rw [abs_le]
  constructor <;> linarith

/-- **Uniform convergence in probability.** Suppose the good events `E β` have complements of
probability tending to 0, and on `E β` the statistic `z q r` is within `C_β r` of `L q` for
every configuration `q` once `r ≤ r₀(β)`. Then `P(|z q r - L q| > η)` tends to 0 as `r → 0`,
uniformly in `q`. -/
theorem uniform_in_prob {ι : Type*} (P : Measure Ω) [IsFiniteMeasure P]
    (z : ι → ℝ → Ω → ℝ) (L : ι → Ω → ℝ) (E : ℝ → Set Ω)
    (hE : Tendsto (fun β => P.real (E β)ᶜ) atTop (𝓝 0))
    (hloc : ∀ β, ∃ C r₀ : ℝ, 0 < r₀ ∧
      ∀ r, 0 < r → r ≤ r₀ → ∀ q, ∀ ω ∈ E β, |z q r ω - L q ω| ≤ C * r)
    {η ε : ℝ} (hη : 0 < η) (hε : 0 < ε) :
    ∃ r₁ > 0, ∀ r, 0 < r → r ≤ r₁ → ∀ q, P.real {ω | η < |z q r ω - L q ω|} ≤ ε := by
  obtain ⟨β, hβ⟩ := (hE.eventually (gt_mem_nhds hε)).exists
  obtain ⟨C, r₀, hr₀, h⟩ := hloc β
  have hC : 0 < |C| + 1 := by positivity
  refine ⟨min r₀ (η / (|C| + 1)), lt_min hr₀ (div_pos hη hC), fun r hr hr1 q => ?_⟩
  refine le_trans (measureReal_mono ?_) hβ.le
  intro ω hω
  by_contra hin
  have hin' : ω ∈ E β := by simpa using hin
  have hb := h r hr (le_trans hr1 (min_le_left _ _)) q ω hin'
  have hr2 : r ≤ η / (|C| + 1) := le_trans hr1 (min_le_right _ _)
  have : C * r < η := by
    calc C * r ≤ |C| * r := mul_le_mul_of_nonneg_right (le_abs_self C) hr.le
      _ < (|C| + 1) * r := by nlinarith
      _ ≤ (|C| + 1) * (η / (|C| + 1)) := mul_le_mul_of_nonneg_left hr2 hC.le
      _ = η := by field_simp
  have hω' : η < |z q r ω - L q ω| := hω
  linarith

/-- **Proposition 5, uniform form.** If every `L q` has distribution function `F`, `F` has a
uniform modulus of continuity, and the hypotheses of `uniform_in_prob` hold, then
`sup_q sup_x |P(z q r ≤ x) - F x| → 0` as `r → 0`. -/
theorem limit_theorem_uniform {ι : Type*} (P : Measure Ω) [IsFiniteMeasure P]
    (z : ι → ℝ → Ω → ℝ) (L : ι → Ω → ℝ) (E : ℝ → Set Ω)
    (hE : Tendsto (fun β => P.real (E β)ᶜ) atTop (𝓝 0))
    (hloc : ∀ β, ∃ C r₀ : ℝ, 0 < r₀ ∧
      ∀ r, 0 < r → r ≤ r₀ → ∀ q, ∀ ω ∈ E β, |z q r ω - L q ω| ≤ C * r)
    (F : ℝ → ℝ) (hF : ∀ q x, P.real {ω | L q ω ≤ x} = F x)
    (hmod : ∀ ε > 0, ∃ η > 0, ∀ x, F (x + η) - F (x - η) ≤ ε) :
    ∀ ε > 0, ∃ r₁ > 0, ∀ r, 0 < r → r ≤ r₁ → ∀ q x, |P.real {ω | z q r ω ≤ x} - F x| ≤ ε := by
  intro ε hε
  obtain ⟨η, hη, hF'⟩ := hmod (ε / 2) (half_pos hε)
  obtain ⟨r₁, hr₁, h⟩ := uniform_in_prob P z L E hE hloc hη (half_pos hε)
  refine ⟨r₁, hr₁, fun r hr hr1 q x => ?_⟩
  have hc := cdf_coupling P (z q r) (L q) hη.le x
  rw [hF, hF, hF] at hc
  linarith [h r hr hr1 q, hF' x]

end Coupling

section HalfNormal

/-- A standard normal variable falls in an interval of length `ℓ` with probability at most `ℓ`
(its density is at most `1/√(2π) < 1`). -/
theorem gaussianReal_Icc_le (a b : ℝ) (hab : a ≤ b) :
    (gaussianReal 0 1).real (Set.Icc a b) ≤ b - a := by
  rw [measureReal_def, gaussianReal_apply_eq_integral 0 one_ne_zero]
  have hle : ∫ x in Set.Icc a b, gaussianPDFReal 0 1 x ≤ ∫ _ in Set.Icc a b, (1 : ℝ) := by
    refine setIntegral_mono_on (integrable_gaussianPDFReal 0 1).integrableOn
      (integrableOn_const (by simp [Real.volume_Icc])) measurableSet_Icc fun x _ => ?_
    rw [gaussianPDFReal_def]
    simp only [NNReal.coe_one, mul_one, sub_zero]
    have he : Real.exp (-x ^ 2 / 2) ≤ 1 := Real.exp_le_one_iff.2 (by nlinarith [sq_nonneg x])
    have hs : 1 ≤ Real.sqrt (2 * Real.pi) := by
      rw [Real.one_le_sqrt]; nlinarith [Real.two_le_pi]
    have hinv : (Real.sqrt (2 * Real.pi))⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hs
    calc (Real.sqrt (2 * Real.pi))⁻¹ * Real.exp (-x ^ 2 / 2) ≤ 1 * 1 :=
          mul_le_mul hinv he (Real.exp_pos _).le zero_le_one
      _ = 1 := one_mul 1
  have hint := integral_gaussianPDFReal_eq_one (μ := 0) (v := 1) one_ne_zero
  rw [ENNReal.toReal_ofReal (setIntegral_nonneg measurableSet_Icc
      fun x _ => gaussianPDFReal_nonneg 0 1 x)]
  simpa [Real.volume_Icc, hab] using hle

/-- **Modulus of the half-normal law.** With `F(x) = P(|N| / (√2 μ) ≤ x)`, `N ~ N(0,1)`:
`F(x + η) - F(x - η) ≤ 4 √2 μ η`. Hence the hypothesis `hmod` of `limit_theorem_uniform` holds
for the limit law of `limit_law`. -/
theorem half_normal_modulus {μ η : ℝ} (hμ : 0 < μ) (hη : 0 ≤ η) (x : ℝ) :
    (gaussianReal 0 1).real {z | |z| / (Real.sqrt 2 * μ) ≤ x + η}
      - (gaussianReal 0 1).real {z | |z| / (Real.sqrt 2 * μ) ≤ x - η}
      ≤ 4 * (Real.sqrt 2 * μ) * η := by
  set s := Real.sqrt 2 * μ with hsdef
  have hs : 0 < s := mul_pos (Real.sqrt_pos.2 two_pos) hμ
  set a := s * (x - η)
  set b := s * (x + η)
  have hab : a ≤ b := by simp only [a, b]; nlinarith
  have hsub : {z | |z| / s ≤ x + η} ⊆
      {z | |z| / s ≤ x - η} ∪ (Set.Icc a b ∪ Set.Icc (-b) (-a)) := by
    intro z hz
    simp only [Set.mem_ofPred_eq, Set.mem_union, Set.mem_Icc] at hz ⊢
    rw [div_le_iff₀ hs] at hz
    by_cases h : |z| / s ≤ x - η
    · exact Or.inl h
    · right
      rw [div_le_iff₀ hs, not_le] at h
      rcases abs_cases z with ⟨hz1, _⟩ | ⟨hz1, _⟩
      · left; constructor <;> nlinarith
      · right; constructor <;> nlinarith
  have hm := measureReal_mono hsub (μ := gaussianReal 0 1)
  have hu := measureReal_union_le (μ := gaussianReal 0 1) {z | |z| / s ≤ x - η}
    (Set.Icc a b ∪ Set.Icc (-b) (-a))
  have hu2 := measureReal_union_le (μ := gaussianReal 0 1) (Set.Icc a b) (Set.Icc (-b) (-a))
  have h1 := gaussianReal_Icc_le a b hab
  have h2 := gaussianReal_Icc_le (-b) (-a) (by linarith)
  have hba : b - a = 2 * s * η := by simp only [a, b]; ring
  linarith

end HalfNormal

end Capri
