/-
The uniform distribution on S³ and its rotation invariance (setup for Proposition 9).

* `E4`, `S3`, `unifS3`: the uniform probability measure on the unit sphere of ℝ⁴, the
  normalised surface measure `volume.toSphere / volume.toSphere univ`.
* `cone_eq`, `unifS3_cone`: for a scale-invariant condition `P`, the uniform measure of
  `{x | P x}` is `vol{y | 0 < ‖y‖ < 1, P y} / vol(ball 0 1)`.
* `unifS3_rot`: the uniform measure is invariant under linear isometries.
* `rot_to_e0`, `rot_to_e01`: a unit vector can be rotated to `e₀`, and an orthonormal pair to
  `(e₀, e₁)` (reflections).
-/
import Mathlib.MeasureTheory.Constructions.HaarToSphere
import Mathlib.MeasureTheory.Measure.Lebesgue.VolumeOfBalls
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace

open MeasureTheory Metric Set
open scoped RealInnerProductSpace Pointwise ENNReal

namespace Capri

/-- ℝ⁴ with its Euclidean structure. -/
abbrev E4 := EuclideanSpace ℝ (Fin 4)

/-- The unit sphere S³. -/
abbrev S3 := sphere (0 : E4) 1

/-- The uniform probability measure on S³. -/
noncomputable def unifS3 : Measure S3 :=
  ((volume : Measure E4).toSphere univ)⁻¹ • (volume : Measure E4).toSphere

theorem vol_ball_E4 : (volume : Measure E4) (ball 0 1) = ENNReal.ofReal (Real.pi ^ 2 / 2) := by
  rw [InnerProductSpace.volume_ball_of_dim_even (k := 2) (by simp)]
  simp

theorem toSphere_univ_E4 :
    (volume : Measure E4).toSphere univ = 4 * ENNReal.ofReal (Real.pi ^ 2 / 2) := by
  rw [Measure.toSphere_apply_univ, vol_ball_E4]; simp

theorem toSphere_univ_ne_zero : (volume : Measure E4).toSphere univ ≠ 0 := by
  rw [toSphere_univ_E4]; simp [Real.pi_pos]

theorem toSphere_univ_ne_top : (volume : Measure E4).toSphere univ ≠ ⊤ := by
  rw [toSphere_univ_E4]; simp [ENNReal.mul_eq_top]

instance : IsProbabilityMeasure unifS3 := by
  constructor
  simp only [unifS3, Measure.smul_apply, smul_eq_mul]
  exact ENNReal.inv_mul_cancel toSphere_univ_ne_zero toSphere_univ_ne_top

/-- A condition invariant under positive scaling. -/
def ScaleInv (P : E4 → Prop) : Prop := ∀ (c : ℝ), 0 < c → ∀ y, P (c • y) ↔ P y

/-- The cone of a scale-invariant condition. -/
theorem cone_eq {P : E4 → Prop} (hP : ScaleInv P) :
    Ioo (0 : ℝ) 1 • ((↑) '' {x : S3 | P x}) = {y : E4 | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ P y} := by
  ext y
  simp only [Set.mem_smul, mem_image, mem_setOf_eq, mem_Ioo, Subtype.exists, mem_sphere_iff_norm,
    sub_zero, exists_and_right, exists_eq_right]
  constructor
  · rintro ⟨r, ⟨hr0, hr1⟩, x, ⟨hx, hPx⟩, rfl⟩
    refine ⟨?_, ?_, (hP r hr0 x).2 hPx⟩ <;>
      simp [norm_smul, abs_of_pos hr0, hx, hr0, hr1]
  · rintro ⟨hy0, hy1, hPy⟩
    refine ⟨‖y‖, ⟨hy0, hy1⟩, ‖y‖⁻¹ • y, ⟨?_, ?_⟩, ?_⟩
    · rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hy0.ne']
    · exact (hP _ (inv_pos.2 hy0) y).2 hPy
    · rw [smul_smul, mul_inv_cancel₀ hy0.ne', one_smul]

/-- **Cone formula.** For a scale-invariant measurable condition, the uniform measure is the
volume of the cone in the unit ball over the volume of the ball. -/
theorem unifS3_cone {P : E4 → Prop} (hP : ScaleInv P) (hm : MeasurableSet {x : S3 | P x}) :
    unifS3 {x : S3 | P x} = (volume : Measure E4) {y | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ P y}
      / ENNReal.ofReal (Real.pi ^ 2 / 2) := by
  simp only [unifS3, Measure.smul_apply, smul_eq_mul]
  rw [Measure.toSphere_apply' _ hm, cone_eq hP, toSphere_univ_E4]
  simp only [finrank_euclideanSpace, Fintype.card_fin, Nat.cast_ofNat]
  rw [ENNReal.mul_inv (by simp) (by simp), div_eq_mul_inv]
  calc _ = (4⁻¹ * 4) * ((volume : Measure E4) {y | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ P y} *
        (ENNReal.ofReal (Real.pi ^ 2 / 2))⁻¹) := by ring
    _ = _ := by
      rw [ENNReal.inv_mul_cancel (by norm_num) (by norm_num), one_mul]; simp only [div_eq_mul_inv]

/-- **Rotation invariance.** -/
theorem unifS3_rot (R : E4 ≃ₗᵢ[ℝ] E4) {P : E4 → Prop} (hP : ScaleInv P)
    (hm : MeasurableSet {x : S3 | P x}) (hmR : MeasurableSet {x : S3 | P (R x)}) :
    unifS3 {x : S3 | P (R x)} = unifS3 {x : S3 | P x} := by
  have hPR : ScaleInv fun y => P (R y) := fun c hc y => by
    simp only [map_smul]; exact hP c hc (R y)
  rw [unifS3_cone hPR hmR, unifS3_cone hP hm]
  congr 1
  have h : {y : E4 | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ P (R y)} = R ⁻¹' {y | 0 < ‖y‖ ∧ ‖y‖ < 1 ∧ P y} := by
    ext y; simp
  rw [h]
  have hR : MeasurePreserving (R.toHomeomorph.toMeasurableEquiv) volume volume :=
    R.measurePreserving
  exact hR.measure_preimage_equiv _

/-- The standard basis vector `e_i` of ℝ⁴. -/
noncomputable def eb (i : Fin 4) : E4 := EuclideanSpace.single i 1

theorem eb_norm (i : Fin 4) : ‖eb i‖ = 1 := by simp [eb]

theorem inner_eb (i : Fin 4) (y : E4) : ⟪eb i, y⟫ = y i := by
  simp [eb, EuclideanSpace.inner_single_left]

/-- A unit vector can be rotated to `e₀`. -/
theorem rot_to_e0 {n : E4} (hn : ‖n‖ = 1) : ∃ R : E4 ≃ₗᵢ[ℝ] E4, R n = eb 0 :=
  ⟨(ℝ ∙ (n - eb 0))ᗮ.reflection, Submodule.reflection_sub (by rw [hn, eb_norm])⟩

/-- An orthonormal pair can be rotated to `(e₀, e₁)`. -/
theorem rot_to_e01 {u₁ u₂ : E4} (h₁ : ‖u₁‖ = 1) (h₂ : ‖u₂‖ = 1) (h₁₂ : ⟪u₁, u₂⟫ = 0) :
    ∃ R : E4 ≃ₗᵢ[ℝ] E4, R u₁ = eb 0 ∧ R u₂ = eb 1 := by
  obtain ⟨R₁, hR₁⟩ := rot_to_e0 h₁
  set v := R₁ u₂
  have hv : ‖v‖ = 1 := by simp [v, h₂]
  have hv0 : ⟪eb 0, v⟫ = 0 := by
    rw [← hR₁]; simp only [v, LinearIsometryEquiv.inner_map_map]; exact h₁₂
  set R₂ := (ℝ ∙ (v - eb 1))ᗮ.reflection
  have he0 : R₂ (eb 0) = eb 0 := by
    apply Submodule.reflection_mem_subspace_eq_self
    rw [Submodule.mem_orthogonal_singleton_iff_inner_right]
    rw [inner_sub_left, real_inner_comm (eb 0) v, hv0, inner_eb]; simp [eb]
  refine ⟨R₁.trans R₂, ?_, ?_⟩
  · simp only [LinearIsometryEquiv.trans_apply, hR₁]; exact he0
  · show (ℝ ∙ (v - eb 1))ᗮ.reflection v = eb 1
    exact Submodule.reflection_sub (by rw [hv, eb_norm])

end Capri
