#!/usr/bin/env bash
# Check the Lean formalization and its failure controls.
#  1. `lake build` must succeed, and no proof may use `sorry`.
#  2. Each control weakens one statement into a false one; Lean must reject it.
# Run from anywhere: bash formal/check.sh   (needs elan; Mathlib v4.35.0 is fetched by lake)
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.elan/bin:$PATH"
fail=0
echo "# commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown); $(lake env lean --version | head -1); Mathlib $(grep -A3 'name = "mathlib"' lakefile.toml | grep rev | cut -d'"' -f2)"
if lake build > /tmp/capri_lake_build.log 2>&1 && ! grep -q "declaration uses 'sorry'\|declaration uses \`sorry\`" /tmp/capri_lake_build.log \
   && ! grep -rn "sorry\|admit" CapriFormal/*.lean > /dev/null; then
  echo "[PASS] lake build: every proof checks, no sorry"
else
  echo "[FAIL] lake build (see /tmp/capri_lake_build.log)"; fail=1
fi
control() {   # name, file, sed expression that makes the statement false, text the false statement contains
  local tmp; tmp=$(mktemp -d)/Mut.lean
  if ! sed "$3" "CapriFormal/$2.lean" > "$tmp" || ! grep -qF -- "$4" "$tmp" || cmp -s "$tmp" "CapriFormal/$2.lean"; then
    echo "[FAIL] control $1: the edit did not apply"; fail=1; return
  fi
  if lake env lean "$tmp" > /dev/null 2>&1; then echo "[FAIL] control $1: Lean accepted a false statement"; fail=1
  else echo "[PASS] control $1: Lean rejects it"; fi
}
control "frame_cr with y/w in place of z/w" Pencil 's|    cr e1 e2 e3 e4 e5 !\[x, y, z, w\] = z / w ∧|    cr e1 e2 e3 e4 e5 ![x, y, z, w] = y / w ∧|' 'cr e1 e2 e3 e4 e5 ![x, y, z, w] = y / w ∧'
control "frame_block_rank with a repeated row" Rank 's|!\[(!\[0, 0, 1 / z, -1 / w\] : Fin 4 → ℝ), !\[0, 1 / y, 0, -1 / w\],|![(![0, 0, 1 / z, -1 / w] : Fin 4 → ℝ), ![0, 0, 1 / z, -1 / w],|' '![(![0, 0, 1 / z, -1 / w] : Fin 4 → ℝ), ![0, 0, 1 / z, -1 / w],'
control "log_abs_change with half the bound" LogBound 's#≤ -log (1 - ε / |b|) := by#≤ -log (1 - ε / |b|) / 2 := by#' '≤ -log (1 - ε / |b|) / 2 := by'
control "many_above with T - k + 2" Quantile 's/    T - k + 1 ≤ (univ.filter (fun t => c < f t)).card := by/    T - k + 2 ≤ (univ.filter (fun t => c < f t)).card := by/' 'T - k + 2 ≤ (univ.filter'
control "hadamard with half the bound" Hadamard 's#det f| ≤ ∏ i, ‖f i‖ := by#det f| ≤ (∏ i, ‖f i‖) / 2 := by#' 'det f| ≤ (∏ i, ‖f i‖) / 2 := by'
control "det_perturb with half the bound" Hadamard 's#det q| ≤ (1 + δ) ^ n - 1 := by#det q| ≤ ((1 + δ) ^ n - 1) / 2 := by#' 'det q| ≤ ((1 + δ) ^ n - 1) / 2 := by'
control "size_bound_interp with k = ceil((T-1)p)" QuantileInterp 's#    (hR : R.card = ⌈((T : ℝ) - 1) \* p⌉₊ + 1) :#    (hR : R.card = ⌈((T : ℝ) - 1) * p⌉₊) :#' '(hR : R.card = ⌈((T : ℝ) - 1) * p⌉₊) :'
control "limit_law with 2 mu in place of sqrt 2 mu" LimitLaw 's#      = (gaussianReal 0 1).map (fun z => |z| / (Real.sqrt 2 \* μ)) := by#      = (gaussianReal 0 1).map (fun z => |z| / (2 * μ)) := by#' '(fun z => |z| / (2 * μ)) := by'
control "small_ball_constant with 16/pi^2" SmallBall 's#      ∫ ρ in (0 : ℝ)..1, 2 \* ρ \* ρ⁻¹) = 32 / π ^ 2 ∧#      ∫ ρ in (0 : ℝ)..1, 2 * ρ * ρ⁻¹) = 16 / π ^ 2 ∧#' '2 * ρ * ρ⁻¹) = 16 / π ^ 2 ∧'
control "unused_eq with [1235] for [1236]" Strata 's#    allBrackets \\ used = {{0, 1, 2, 5},#    allBrackets \\ used = {{0, 1, 2, 4},#' 'allBrackets \ used = {{0, 1, 2, 4},'
control "rank_all_pencils with 3n - 14" RankFull 's#Function.Injective t} → ℝ)) = 3 \* n - 15 := by#Function.Injective t} → ℝ)) = 3 * n - 14 := by#' 'Function.Injective t} → ℝ)) = 3 * n - 14 := by'
control "orbitMap_range_finrank with n + 16" RankFull 's#    finrank ℝ (LinearMap.range (orbitMap P)) = n + 15 := by#    finrank ℝ (LinearMap.range (orbitMap P)) = n + 16 := by#' '(LinearMap.range (orbitMap P)) = n + 16 := by'
control "separation_signed with points k >= 6 only" Separation 's#^    (∀ k : Fin n, 5 ≤ k.val → ∀ j : Fin 3,$#    (∀ k : Fin n, 6 ≤ k.val → ∀ j : Fin 3,#' '    (∀ k : Fin n, 6 ≤ k.val → ∀ j : Fin 3,'
control "no_reverse with [1236] = 1" Kappa 's#∧ brf 0 1 2 5 P0 = 0 := by#∧ brf 0 1 2 5 P0 = 1 := by#' '∧ brf 0 1 2 5 P0 = 1 := by'
control "gaussian_abs_gt with 1/12" NoiseLaw 's#    (𝒩).real {x | b < |x|} ≤ 1 / 6 := by#    (𝒩).real {x | b < |x|} ≤ 1 / 12 := by#' '(𝒩).real {x | b < |x|} ≤ 1 / 12 := by'
control "rate_used with c/d^2" Rate 's#      ∀ᶠ d in 𝓝\[>\] 0, c / d ≤ kappaJ (Pd d)#      ∀ᶠ d in 𝓝[>] 0, c / d ^ 2 ≤ kappaJ (Pd d)#' 'c / d ^ 2 ≤ kappaJ (Pd d)'
control "taylor_uniform with a cubic remainder" Taylor 's#      |f (q + h) - f q - fderiv ℝ f q h| ≤ C \* ‖h‖ ^ 2 := by#      |f (q + h) - f q - fderiv ℝ f q h| ≤ C * ‖h‖ ^ 3 := by#' '|f (q + h) - f q - fderiv ℝ f q h| ≤ C * ‖h‖ ^ 3 := by'
control "size_power_limit with P(Z > c)" ScaleFamily 's#(𝓝 (P.real {ω | c / f < Z ω})) := by#(𝓝 (P.real {ω | c < Z ω})) := by#' '(𝓝 (P.real {ω | c < Z ω})) := by'
control "euclid_le_sup with factor 1" Pivotal 's#‖WithLp.toLp 2 x‖ ≤ 2 \* ‖x‖ := by#‖WithLp.toLp 2 x‖ ≤ 1 * ‖x‖ := by#' '‖WithLp.toLp 2 x‖ ≤ 1 * ‖x‖ := by'
control "toSphere_univ_E4 with 2 in place of 4" Sphere 's#    (volume : Measure E4).toSphere univ = 4 \* ENNReal.ofReal (Real.pi ^ 2 / 2) := by#    (volume : Measure E4).toSphere univ = 2 * ENNReal.ofReal (Real.pi ^ 2 / 2) := by#' 'toSphere univ = 2 * ENNReal.ofReal (Real.pi ^ 2 / 2) := by'
control "unif_abs_inner_lt with Psi(a)/2" SphereLaw 's#    unifS3 {x : S3 | |⟪n, (x : E4)⟫| < a} = ENNReal.ofReal (Psi a) := by#    unifS3 {x : S3 | |⟪n, (x : E4)⟫| < a} = ENNReal.ofReal (Psi a / 2) := by#' '< a} = ENNReal.ofReal (Psi a / 2) := by'
control "vol_coneII with pi^2/4" SphereLaw 's#    volume (coneII b) = ENNReal.ofReal (π ^ 2 / 2 \* b) := by#    volume (coneII b) = ENNReal.ofReal (π ^ 2 / 4 * b) := by#' 'volume (coneII b) = ENNReal.ofReal (π ^ 2 / 4 * b) := by'
control "small_ball_limit with 16/pi^2" SphereSmallBall 's#    Tendsto (fun d => unif4.real (SB d) / d) (𝓝\[>\] 0) (𝓝 (32 / π ^ 2)) := by#    Tendsto (fun d => unif4.real (SB d) / d) (𝓝[>] 0) (𝓝 (16 / π ^ 2)) := by#' '(𝓝 (16 / π ^ 2)) := by'
[ $fail -eq 0 ] && echo "ALL LEAN CHECKS PASSED" || { echo "LEAN CHECKS FAILED"; exit 1; }
