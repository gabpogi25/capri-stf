/-
The Taylor step behind Theorem 4.1 (first-order scale family) and Proposition 4.4.

* `taylor_uniform`: a function that is C² on an open set `U` has a second-order Taylor
  remainder `|f(q + h) - f(q) - Df(q) h| ≤ C ‖h‖²`, uniformly for `q` in a compact `K ⊆ U` and
  `‖h‖ ≤ δ₀` (mean value theorem, twice).
* `crAt_contDiffOn`, `logcr_contDiffOn`: `log|CR_t|` is smooth where its four brackets are nonzero.
* `scale_family`: Theorem 4.1 for one pencil. With measurements `q + r u` and `q + r v`, each
  followed by any gauge, `D_t = |log|CR_t(A)| - log|CR_t(B)||` satisfies
  `|D_t - r |Dφ_t(q)(u - v)|| ≤ C r²`, uniformly over a compact set of configurations on which
  the brackets of `t` are nonzero, `‖u‖, ‖v‖ ≤ B` and `0 ≤ r ≤ r₀`.
-/
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.MetricSpace.Thickening
import CapriFormal.RankFull

open Filter Topology Metric

namespace Capri

/-- **Uniform second-order Taylor bound** on a compact subset of an open set where `f` is C². -/
theorem taylor_uniform {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [ProperSpace E]
    {f : E → ℝ} {U K : Set E} (hU : IsOpen U) (hK : IsCompact K) (hKU : K ⊆ U)
    (hf : ContDiffOn ℝ 2 f U) :
    ∃ C δ₀ : ℝ, 0 ≤ C ∧ 0 < δ₀ ∧ ∀ q ∈ K, ∀ h : E, ‖h‖ ≤ δ₀ →
      |f (q + h) - f q - fderiv ℝ f q h| ≤ C * ‖h‖ ^ 2 := by
  obtain ⟨δ₀, hδ₀, hsub⟩ := hK.exists_cthickening_subset_open hU hKU
  have hC1 : ContDiffOn ℝ 1 (fderiv ℝ f) U := hf.fderiv_of_isOpen hU (by norm_num)
  have hcont2 : ContinuousOn (fderiv ℝ (fderiv ℝ f)) U :=
    hC1.continuousOn_fderiv_of_isOpen hU le_rfl
  have hnc : ContinuousOn (fun x => ‖fderiv ℝ (fderiv ℝ f) x‖) (cthickening δ₀ K) := by
    exact ContinuousOn.norm (f := fderiv ℝ (fderiv ℝ f)) (hcont2.mono hsub)
  obtain ⟨M, hM'⟩ := (hK.cthickening (r := δ₀)).exists_bound_of_continuousOn hnc
  have hM : ∀ x ∈ cthickening δ₀ K, ‖fderiv ℝ (fderiv ℝ f) x‖ ≤ M := fun x hx => by
    have := hM' x hx
    rw [Real.norm_eq_abs] at this
    exact le_trans (le_abs_self _) this
  refine ⟨max M 0, δ₀, le_max_right _ _, hδ₀, fun q hq h hh => ?_⟩
  set r := ‖h‖ with hr
  have hballK : closedBall q r ⊆ cthickening δ₀ K :=
    (closedBall_subset_closedBall hh).trans (closedBall_subset_cthickening hq δ₀)
  have hball : closedBall q r ⊆ U := hballK.trans hsub
  have hdiff : ∀ y ∈ U, DifferentiableAt ℝ f y := fun y hy =>
    (hf.differentiableOn (by norm_num)).differentiableAt (hU.mem_nhds hy)
  have hdiff2 : ∀ y ∈ U, DifferentiableAt ℝ (fderiv ℝ f) y := fun y hy =>
    (hC1.differentiableOn one_ne_zero).differentiableAt (hU.mem_nhds hy)
  set L := fderiv ℝ f q with hL
  have hq : q ∈ closedBall q r := mem_closedBall_self (norm_nonneg h)
  -- Df is (max M 0)-Lipschitz on the ball
  have hDf : ∀ y ∈ closedBall q r, ‖fderiv ℝ f y - L‖ ≤ max M 0 * r := by
    intro y hy
    have h1 := Convex.norm_image_sub_le_of_norm_fderiv_le (f := fderiv ℝ f) (C := max M 0)
      (fun x hx => hdiff2 x (hball hx))
      (fun x hx => le_trans (hM x (hballK hx)) (le_max_left _ _)) (convex_closedBall q r) hq hy
    refine le_trans h1 (mul_le_mul_of_nonneg_left ?_ (le_max_right _ _))
    rw [← dist_eq_norm]; exact hy
  -- ψ(y) = f y - L (y - q) has derivative Df(y) - L
  set ψ : E → ℝ := fun y => f y - (L y - L q) with hψ
  have hψd : ∀ y ∈ closedBall q r, HasFDerivAt ψ (fderiv ℝ f y - L) y := by
    intro y hy
    have h1 := (hdiff y (hball hy)).hasFDerivAt
    have h2 : HasFDerivAt (fun y : E => L y - L q) L y := L.hasFDerivAt.sub_const (L q)
    exact h1.sub h2
  have hmvt := Convex.norm_image_sub_le_of_norm_fderiv_le (f := ψ) (C := max M 0 * r)
    (fun y hy => (hψd y hy).differentiableAt)
    (fun y hy => by rw [(hψd y hy).fderiv]; exact hDf y hy) (convex_closedBall q r) hq
    (show q + h ∈ closedBall q r by simp [hr])
  have e : ψ (q + h) - ψ q = f (q + h) - f q - L h := by
    simp only [hψ, map_add]; ring
  rw [e, Real.norm_eq_abs, add_sub_cancel_left] at hmvt
  calc |f (q + h) - f q - L h| ≤ max M 0 * r * ‖h‖ := hmvt
    _ = max M 0 * ‖h‖ ^ 2 := by rw [hr]; ring

variable {n : ℕ}

/-- The open set where the four brackets of the pencil `t` are nonzero. -/
def Ut (t : Fin 6 → Fin n) : Set (Config n) :=
  {Q | br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 4)) ≠ 0 ∧ br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 5)) ≠ 0
    ∧ br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 5)) ≠ 0 ∧ br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 4)) ≠ 0}

theorem br_contDiff (a b c d : Fin n) :
    ContDiff ℝ 2 (fun Q : Config n => br (Q a) (Q b) (Q c) (Q d)) := by
  simp only [br_expand]
  fun_prop

theorem Ut_isOpen (t : Fin 6 → Fin n) : IsOpen (Ut t) := by
  have c := fun a b c d => (br_contDiff (n := n) a b c d).continuous
  exact (isOpen_ne_fun (c _ _ _ _) continuous_const).and
    ((isOpen_ne_fun (c _ _ _ _) continuous_const).and
    ((isOpen_ne_fun (c _ _ _ _) continuous_const).and
    (isOpen_ne_fun (c _ _ _ _) continuous_const)))

/-- `log|CR_t|` is C² (indeed smooth) where the four brackets of `t` are nonzero. -/
theorem logcr_contDiffOn (t : Fin 6 → Fin n) :
    ContDiffOn ℝ 2 (fun Q => Real.log (crAt t Q)) (Ut t) := by
  intro Q hQ
  obtain ⟨h1, h2, h3, h4⟩ := hQ
  have hden : br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 5)) * br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 4))
      ≠ 0 := mul_ne_zero h3 h4
  have hcr : crAt t Q ≠ 0 := by
    simp only [crAt, cr]; exact div_ne_zero (mul_ne_zero h1 h2) hden
  have hc : ContDiffAt ℝ 2 (crAt t) Q := by
    have e : crAt t = fun Q : Config n =>
        br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 4)) * br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 5)) /
          (br (Q (t 0)) (Q (t 1)) (Q (t 2)) (Q (t 5)) * br (Q (t 0)) (Q (t 1)) (Q (t 3)) (Q (t 4))) :=
      rfl
    rw [e]
    exact (((br_contDiff _ _ _ _).contDiffAt).mul (br_contDiff _ _ _ _).contDiffAt).div
      (((br_contDiff _ _ _ _).contDiffAt).mul (br_contDiff _ _ _ _).contDiffAt) hden
  exact (hc.log hcr).contDiffWithinAt

theorem crAt_act' (t : Fin 6 → Fin n) (A : Matrix (Fin 4) (Fin 4) ℝ) (hA : A.det ≠ 0)
    (l : Fin n → ℝ) (hl : ∀ k, l k ≠ 0) (P : Config n) : crAt t (act A l P) = crAt t P := by
  simp only [crAt, act]
  rw [cr_gauge A hA _ _ _ _ _ _ (hl _) (hl _) (hl _) (hl _) (hl _) (hl _)]

/-- **Theorem 4.1 (first-order scale family), one pencil.** On a compact set `K` of
configurations where the brackets of `t` are nonzero, with noise `‖u‖, ‖v‖ ≤ B`, measurements
`q + r u` and `q + r v` each followed by any gauge, and `0 ≤ r ≤ r₀`:
`|D_t - r |Dφ_t(q)(u - v)|| ≤ C r²`, `φ_t = log|CR_t|`, uniformly in `q ∈ K`. -/
theorem scale_family (t : Fin 6 → Fin n) {K : Set (Config n)} (hK : IsCompact K)
    (hKU : K ⊆ Ut t) {B : ℝ} (hB : 0 ≤ B) :
    ∃ C r₀ : ℝ, 0 ≤ C ∧ 0 < r₀ ∧ ∀ q ∈ K, ∀ u v : Config n, ‖u‖ ≤ B → ‖v‖ ≤ B →
      ∀ r, 0 ≤ r → r ≤ r₀ → ∀ (A₁ A₂ : Matrix (Fin 4) (Fin 4) ℝ) (l₁ l₂ : Fin n → ℝ),
      A₁.det ≠ 0 → A₂.det ≠ 0 → (∀ k, l₁ k ≠ 0) → (∀ k, l₂ k ≠ 0) →
      abs (|Real.log (crAt t (act A₁ l₁ (q + r • u))) - Real.log (crAt t (act A₂ l₂ (q + r • v)))|
        - r * |fderiv ℝ (fun Q => Real.log (crAt t Q)) q (u - v)|) ≤ C * r ^ 2 := by
  obtain ⟨C₀, δ₀, hC₀, hδ₀, hT⟩ := taylor_uniform (Ut_isOpen t) hK hKU (logcr_contDiffOn t)
  refine ⟨2 * C₀ * B ^ 2, δ₀ / (B + 1), by positivity, by positivity, ?_⟩
  intro q hq u v hu hv r hr0 hr A₁ A₂ l₁ l₂ hA₁ hA₂ hl₁ hl₂
  rw [crAt_act' t A₁ hA₁ l₁ hl₁, crAt_act' t A₂ hA₂ l₂ hl₂]
  set φ := fun Q => Real.log (crAt t Q)
  set L := fderiv ℝ φ q
  have hsmall : ∀ w : Config n, ‖w‖ ≤ B → ‖r • w‖ ≤ δ₀ := by
    intro w hw
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hr0]
    calc r * ‖w‖ ≤ δ₀ / (B + 1) * B := mul_le_mul hr hw (norm_nonneg _) (by positivity)
      _ ≤ δ₀ := by
        rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]; nlinarith
  have hsq : ∀ w : Config n, ‖w‖ ≤ B → ‖r • w‖ ^ 2 ≤ r ^ 2 * B ^ 2 := by
    intro w hw
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hr0, mul_pow]
    exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (norm_nonneg _) hw 2) (by positivity)
  have e1 := hT q hq (r • u) (hsmall u hu)
  have e2 := hT q hq (r • v) (hsmall v hv)
  have hLu : L (r • u) - L (r • v) = r * L (u - v) := by
    rw [map_smul, map_smul, map_sub, smul_eq_mul, smul_eq_mul]; ring
  have key : |(φ (q + r • u) - φ (q + r • v)) - r * L (u - v)| ≤ 2 * C₀ * B ^ 2 * r ^ 2 := by
    have : (φ (q + r • u) - φ (q + r • v)) - r * L (u - v) =
        (φ (q + r • u) - φ q - L (r • u)) - (φ (q + r • v) - φ q - L (r • v)) := by
      rw [← hLu]; ring
    rw [this]
    calc _ ≤ |φ (q + r • u) - φ q - L (r • u)| + |φ (q + r • v) - φ q - L (r • v)| :=
          abs_sub _ _
      _ ≤ C₀ * ‖r • u‖ ^ 2 + C₀ * ‖r • v‖ ^ 2 := add_le_add e1 e2
      _ ≤ C₀ * (r ^ 2 * B ^ 2) + C₀ * (r ^ 2 * B ^ 2) := by
          gcongr
          · exact hsq u hu
          · exact hsq v hv
      _ = 2 * C₀ * B ^ 2 * r ^ 2 := by ring
  have habs : r * |L (u - v)| = |r * L (u - v)| := by rw [abs_mul, abs_of_nonneg hr0]
  rw [habs]
  exact le_trans (abs_abs_sub_abs_le_abs_sub _ _) key

/-- **Uniform Lipschitz bound for the derivative** on a compact subset of an open set where `f`
is C²: `‖Df(x) - Df(q)‖ ≤ M ‖x - q‖` for `q ∈ K`, `‖x - q‖ ≤ δ₀`. -/
theorem fderiv_lip_uniform {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [ProperSpace E]
    {f : E → ℝ} {U K : Set E} (hU : IsOpen U) (hK : IsCompact K) (hKU : K ⊆ U)
    (hf : ContDiffOn ℝ 2 f U) :
    ∃ M δ₀ : ℝ, 0 ≤ M ∧ 0 < δ₀ ∧ ∀ q ∈ K, ∀ x : E, ‖x - q‖ ≤ δ₀ →
      ‖fderiv ℝ f x - fderiv ℝ f q‖ ≤ M * ‖x - q‖ := by
  obtain ⟨δ₀, hδ₀, hsub⟩ := hK.exists_cthickening_subset_open hU hKU
  have hC1 : ContDiffOn ℝ 1 (fderiv ℝ f) U := hf.fderiv_of_isOpen hU (by norm_num)
  have hcont2 : ContinuousOn (fderiv ℝ (fderiv ℝ f)) U :=
    hC1.continuousOn_fderiv_of_isOpen hU le_rfl
  have hnc : ContinuousOn (fun x => ‖fderiv ℝ (fderiv ℝ f) x‖) (cthickening δ₀ K) := by
    exact ContinuousOn.norm (f := fderiv ℝ (fderiv ℝ f)) (hcont2.mono hsub)
  obtain ⟨M, hM'⟩ := (hK.cthickening (r := δ₀)).exists_bound_of_continuousOn hnc
  have hM : ∀ x ∈ cthickening δ₀ K, ‖fderiv ℝ (fderiv ℝ f) x‖ ≤ M := fun x hx => by
    have := hM' x hx
    rw [Real.norm_eq_abs] at this
    exact le_trans (le_abs_self _) this
  refine ⟨max M 0, δ₀, le_max_right _ _, hδ₀, fun q hq x hx => ?_⟩
  have hballK : closedBall q δ₀ ⊆ cthickening δ₀ K := closedBall_subset_cthickening hq δ₀
  have hdiff2 : ∀ y ∈ U, DifferentiableAt ℝ (fderiv ℝ f) y := fun y hy =>
    (hC1.differentiableOn one_ne_zero).differentiableAt (hU.mem_nhds hy)
  exact Convex.norm_image_sub_le_of_norm_fderiv_le (f := fderiv ℝ f) (C := max M 0)
    (fun y hy => hdiff2 y (hsub (hballK hy)))
    (fun y hy => le_trans (hM y (hballK hy)) (le_max_left _ _)) (convex_closedBall q δ₀)
    (mem_closedBall_self hδ₀.le) (by rw [mem_closedBall, dist_eq_norm]; exact hx)

end Capri
