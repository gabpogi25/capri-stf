"""
tests/test_coplanarity.py

Tests for algorithms/degeneracy/coplanarity.py and the scoring and statistics helpers of
experiments/run_degeneracy_matched.py (pre-registration docs/prereg/degeneracy_matched.md,
Sections 2, 6.1 and 10). Each invariance test also runs a negative control showing that the
same change moves a quantity that is NOT invariant, so the test can fail for the reason it tests.
Seeds are small literals outside every experiment band.
"""

import multiprocessing
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import run_degeneracy_matched as exp14  # noqa: E402
import run_synthetic_baselines as rsb  # noqa: E402
from algorithms.degeneracy import coplanarity as cop  # noqa: E402
from reconstruction.triangulation import (  # noqa: E402
    estimate_fundamental_matrix_eight_point, estimate_projection_matrices_from_fundamental, triangulate_point_cloud)

FOCAL = np.array([[800.0, 0, 640], [0, 800.0, 480], [0, 0, 1]])
ROTATION = np.array([[np.cos(0.3), 0, np.sin(0.3)], [0, 1, 0], [-np.sin(0.3), 0, np.cos(0.3)]])
TRANSLATION = np.array([-2.0, 0.2, 0.5])
CAMERA_1_PX = FOCAL @ np.hstack([np.eye(3), np.zeros((3, 1))])
CAMERA_2_PX = FOCAL @ np.hstack([ROTATION, TRANSLATION[:, None]])


def exact_scene(seed: int, n_planar: int, n_points: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free pixels of n_points points, the first n_planar exactly on one plane (make_scene geometry)."""
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    basis_u = np.cross(normal, [0.0, 0.0, 1.0])
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(normal, basis_u)
    coefficients = rng.uniform(-3, 3, (n_planar, 2))
    planar = np.array([0.0, 0.0, 8.0]) + coefficients[:, :1] * basis_u + coefficients[:, 1:] * basis_v
    generic = np.column_stack([rng.uniform(-3, 3, n_points - n_planar), rng.uniform(-2, 2, n_points - n_planar),
                               rng.uniform(4, 12, n_points - n_planar)])
    world = np.column_stack([np.vstack([planar, generic]), np.ones(n_points)])
    project = lambda camera: (world @ camera.T)[:, :2] / (world @ camera.T)[:, 2:]
    return project(CAMERA_1_PX), project(CAMERA_2_PX)


def eight_point_perturbed(view_1: np.ndarray, view_2: np.ndarray, step: float = 1e-3, charts=None) -> np.ndarray:
    """M8 reconstructions (B, 4, 8) of the perturbed copies in the Hartley charts (or given charts)."""
    chart_1, chart_2 = charts or (rsb.hartley_normalize(view_1)[1], rsb.hartley_normalize(view_2)[1])
    return cop.batch_eight_point_reconstruction(*cop.perturbed_correspondences(view_1, view_2, step), chart_1, chart_2)[0]


def first_six_distance(perturbed: np.ndarray, view_1: np.ndarray, base=None, step: float = 1e-3,
                       sigma: float = 0.5) -> tuple[float, float]:
    """(T, ||g||^2) of the subset of the first six points, base triple largest-area unless given."""
    subset = np.arange(6)[None]
    base = cop.largest_triangle_base_triples(view_1, subset) if base is None else np.array([base])
    constraints = cop.bracket_constraints(perturbed, subset, base)
    distances, status = cop.sampson_distances(constraints, step, sigma)
    assert status[0] == cop.STATUS_FULL_RANK
    return float(distances[0]), float(np.sum(constraints[0, 0] ** 2))


def noisy(seed: int, n_planar: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    view_1, view_2 = exact_scene(seed, n_planar)
    noise = np.random.default_rng(seed + 1).standard_normal((2, 8, 2)) * sigma
    return view_1 + noise[0], view_2 + noise[1]


def constraints_from(value: np.ndarray, jacobian: np.ndarray, step: float = 1e-3) -> np.ndarray:
    """Perturbed-copy constraints (1 + 2N, 1, r) whose central difference is exactly `jacobian` (r, N)."""
    plus = value[None] + step * jacobian.T
    minus = value[None] - step * jacobian.T
    return np.concatenate([value[None], plus, minus])[:, None, :]


# ------------------------------------------------------------------ geometry and invariance

def test_exact_coplanar_subset_has_zero_distance_and_generic_does_not():
    view_1, view_2 = exact_scene(3, 6)
    profile = cop.stratum_distance_profile(eight_point_perturbed(view_1, view_2), view_1, (6,), 1e-3, 0.5)
    distances = profile[6][0]
    assert distances[0] < 1e-8                                  # subset (0..5) is the coplanar one
    assert np.min(distances[1:]) > 1.0


@pytest.mark.parametrize("change", ["frame", "rescale", "base_triple"])
def test_distance_first_order_invariant_inside_perturbed_reconstructions(change):
    # seed 6 gives a well-conditioned 8-point system; on badly conditioned ones (sigma_8/sigma_1 ~ 1e-4)
    # the first-order regime needs far smaller noise (module docstring, item (7))
    view_1, view_2 = noisy(6, 6, 0.01)
    perturbed = eight_point_perturbed(view_1, view_2)
    copies_1, copies_2 = cop.perturbed_correspondences(view_1, view_2, 1e-3)
    data = np.concatenate([copies_1.reshape(len(copies_1), -1), copies_2.reshape(len(copies_2), -1)], axis=1)
    changed, base = perturbed, None
    if change == "frame":       # constant A applied before unit normalisation: per-point factors depend on u
        frame = np.random.default_rng(7).standard_normal((4, 4)) + 2 * np.eye(4)
        changed = frame @ perturbed
        changed /= np.linalg.norm(changed, axis=1, keepdims=True)
    elif change == "rescale":   # smooth u-dependent per-point factors
        weights = np.random.default_rng(8).standard_normal((32, 8)) / 300.0
        changed = perturbed * np.exp(np.sin(data @ weights))[:, None, :]
    else:                       # any other non-collinear triple of the subset
        largest = tuple(cop.largest_triangle_base_triples(view_1, np.arange(6)[None])[0])
        base = (0, 2, 4) if largest != (0, 2, 4) else (1, 3, 5)
    original, original_norm = first_six_distance(perturbed, view_1, sigma=0.01)
    moved, moved_norm = first_six_distance(changed, view_1, base=base, sigma=0.01)
    assert abs(moved / original - 1) < 5e-3
    assert abs(moved_norm / original_norm - 1) > 0.1          # negative control: ||g||^2 is not invariant


def test_affine_representatives_change_distance_beyond_first_order():
    # scope of item (5)(ii) (prereg post-hoc checker note): the invariance is first order only. On a
    # badly conditioned configuration, affine representatives X/X_4 move T by > 20% at 1e-3 px; the same
    # change is within 1e-3 on the well-conditioned seed used by the invariance test above
    ratios = []
    for seed in (12, 6):
        view_1, view_2 = noisy(seed, 6, 1e-3)
        perturbed = eight_point_perturbed(view_1, view_2)
        unit, _ = first_six_distance(perturbed, view_1, sigma=1e-3)
        affine, _ = first_six_distance(perturbed / perturbed[:, 3:4, :], view_1, sigma=1e-3)
        ratios.append(affine / unit)
    assert abs(ratios[0] - 1) > 0.2
    assert abs(ratios[1] - 1) < 1e-3


def test_distance_changes_with_image_chart():
    view_1, view_2 = noisy(12, 6, 5e-3)
    hartley = (rsb.hartley_normalize(view_1)[1], rsb.hartley_normalize(view_2)[1])
    affine = np.array([[3.0, 0.7, 0.2], [0.0, 0.4, -0.3], [0, 0, 1.0]])
    in_hartley, _ = first_six_distance(eight_point_perturbed(view_1, view_2, charts=hartley), view_1)
    in_affine, _ = first_six_distance(eight_point_perturbed(view_1, view_2, charts=(affine @ hartley[0], affine @ hartley[1])), view_1)
    assert abs(in_affine / in_hartley - 1) > 1e-2


def test_linear_eight_point_fundamental_is_not_rank_two():
    view_1, view_2 = noisy(13, 0, 0.5)
    design = cop.eight_point_design_matrix(cop.apply_image_chart(view_1, rsb.hartley_normalize(view_1)[1])[None],
                                           cop.apply_image_chart(view_2, rsb.hartley_normalize(view_2)[1])[None])
    linear = cop.linear_fundamental_matrix(design)[0]
    assert abs(np.linalg.det(linear)) / np.linalg.norm(linear) ** 3 > 1e-6
    assert abs(np.linalg.det(cop.project_to_rank_two(linear))) < 1e-12


@pytest.mark.parametrize("n_planar,expect_rank", [(6, 8), (7, 7), (8, 6)])
def test_rank_of_noise_free_eight_point_system(n_planar, expect_rank):
    view_1, view_2 = exact_scene(17, n_planar)
    design = cop.eight_point_design_matrix(cop.apply_image_chart(view_1, rsb.hartley_normalize(view_1)[1])[None],
                                           cop.apply_image_chart(view_2, rsb.hartley_normalize(view_2)[1])[None])[0]
    singular = np.linalg.svd(design, compute_uv=False)
    assert singular[expect_rank - 1] / singular[0] > 1e-6
    assert expect_rank == 8 or singular[expect_rank] / singular[0] < 1e-10


def test_gap_frame_bound_and_monotonicity():
    rng = np.random.default_rng(19)
    for _ in range(20):
        points = rng.standard_normal((4, 8))
        frame = rng.standard_normal((4, 4))
        condition = np.linalg.cond(frame)
        gaps = [cop.coplanarity_gap(points, m) for m in range(4, 9)]
        assert np.all(np.diff(gaps) >= -1e-15)
        for m in range(4, 9):
            moved = cop.coplanarity_gap(frame @ points, m)
            assert gaps[m - 4] / condition * (1 - 1e-12) <= moved <= gaps[m - 4] * condition * (1 + 1e-12)


def test_batched_reconstruction_agrees_with_repository_helpers():
    view_1, view_2 = noisy(23, 3, 0.5)
    chart_1, chart_2 = rsb.hartley_normalize(view_1)[1], rsb.hartley_normalize(view_2)[1]
    points, fundamental = cop.batch_eight_point_reconstruction(view_1[None], view_2[None], chart_1, chart_2)
    norm_1, norm_2 = rsb.hartley_normalize(view_1)[0], rsb.hartley_normalize(view_2)[0]
    camera_1, camera_2 = estimate_projection_matrices_from_fundamental(fundamental[0], norm_1, norm_2)
    reference = triangulate_point_cloud([norm_1, norm_2], np.stack([camera_1, camera_2]))
    reference /= np.linalg.norm(reference, axis=0)
    assert np.allclose(np.abs(np.sum(reference * points[0], axis=0)), 1.0, atol=1e-9)
    fixed = cop.batch_fixed_camera_triangulation(view_1[None], view_2[None], chart_1, chart_2, camera_1, camera_2)[0]
    assert np.allclose(np.abs(np.sum(reference * fixed, axis=0)), 1.0, atol=1e-9)
    pixel_fundamental = chart_2.T @ fundamental[0] @ chart_1
    helper = estimate_fundamental_matrix_eight_point(view_1, view_2)
    assert np.isclose(abs(np.sum(helper * pixel_fundamental / np.linalg.norm(pixel_fundamental))), 1.0, atol=1e-9)


def test_chi2_mean_with_fixed_cameras_detects_jacobian_scale():
    view_1, view_2 = exact_scene(29, 6)
    rng, sigma, distances = np.random.default_rng(31), 0.01, []
    for _ in range(400):
        noisy_1, noisy_2 = view_1 + sigma * rng.standard_normal((8, 2)), view_2 + sigma * rng.standard_normal((8, 2))
        perturbed = cop.batch_fixed_camera_triangulation(*cop.perturbed_correspondences(noisy_1, noisy_2, 1e-4), np.eye(3),
                                                         np.eye(3), CAMERA_1_PX, CAMERA_2_PX)
        distances.append(first_six_distance(perturbed, noisy_1, step=1e-4, sigma=sigma)[0])
    assert 2.5 < np.mean(distances) < 3.5                       # chi2_3 mean; a factor 2 in J gives 0.75 or 12


@pytest.mark.parametrize("n_planar", [0, 3, 5, 6])
def test_step_sizes_agree_for_at_most_six_planar_points(n_planar):
    view_1, view_2, planar = rsb.make_scene(np.random.default_rng(37))
    sample = np.r_[np.flatnonzero(planar)[:n_planar], np.flatnonzero(~planar)[:8 - n_planar]]
    coarse = cop.stratum_distance_profile(eight_point_perturbed(view_1[sample], view_2[sample], 1e-3), view_1[sample], (6,), 1e-3, 0.5)
    fine = cop.stratum_distance_profile(eight_point_perturbed(view_1[sample], view_2[sample], 1e-4), view_1[sample], (6,), 1e-4, 0.5)
    assert np.allclose(coarse[6][0], fine[6][0], rtol=exp14.STEP_AGREEMENT_RTOL)


# ------------------------------------------------------------------ prereg Section 6.1 cases

def test_singular_jacobian_infeasible_gives_inf_and_feasible_gives_pseudo_inverse():
    jacobian = np.array([[1.0, 0.0], [2.0, 0.0]])
    infeasible, status = cop.sampson_distances(constraints_from(np.array([1.0, 0.0]), jacobian), 1e-3, 0.5)
    assert np.isinf(infeasible[0]) and status[0] == cop.STATUS_INFEASIBLE
    value = np.array([1.0, 2.0])
    feasible, status = cop.sampson_distances(constraints_from(value, jacobian), 1e-3, 0.5)
    expected = value @ np.linalg.pinv(jacobian @ jacobian.T) @ value / 0.25
    assert status[0] == cop.STATUS_FEASIBLE_PINV and np.isclose(feasible[0], expected, rtol=1e-6)
    zero_j, status = cop.sampson_distances(constraints_from(np.zeros(2), np.zeros((2, 2))), 1e-3, 0.5)
    assert zero_j[0] == 0.0 and status[0] == cop.STATUS_FEASIBLE_PINV
    full = np.array([[1.0, 0.5, 0.0], [0.2, 1.0, 3.0]])
    distance, status = cop.sampson_distances(constraints_from(value, full), 1e-3, 0.5)
    assert status[0] == cop.STATUS_FULL_RANK and np.isclose(distance[0], value @ np.linalg.solve(full @ full.T, value) / 0.25)


def test_nonfinite_subsets_are_left_out_and_all_nan_maps_to_least():
    good = constraints_from(np.array([0.1, 0.2]), np.eye(2))
    bad = constraints_from(np.array([np.nan, 0.2]), np.eye(2))
    distances, status = cop.sampson_distances(np.concatenate([good, bad], axis=1), 1e-3, 1.0)
    assert np.isnan(distances[1]) and status[1] == cop.STATUS_NONFINITE
    assert cop.coplanarity_score(distances, status) == pytest.approx(-np.log10(distances[0]))
    all_nan = cop.coplanarity_score(np.array([np.nan, np.nan]), np.array([1, 1]))
    assert np.isnan(all_nan)
    assert exp14.map_nonfinite(np.array([all_nan]), "primary")[0] == exp14.LEAST
    assert exp14.map_nonfinite(np.array([all_nan]), "opposite")[0] == exp14.MOST
    assert cop.coplanarity_score(np.array([0.0, 1.0]), np.zeros(2, int)) == np.inf
    infeasible = cop.coplanarity_score(np.array([np.inf]), np.array([cop.STATUS_INFEASIBLE]))
    assert infeasible == -np.inf and cop.coplanarity_score(np.array([np.inf]), np.array([3]), 0.0) == np.inf


def test_point_mapped_to_infinity_moves_2d_scores_toward_least():
    view_1, view_2 = noisy(41, 6, 0.5)
    homography = np.eye(3)
    homography[2] = [1.0, 0.0, -view_1[0, 0]]                  # third row vanishes at point 0
    assert np.isinf(rsb.transfer_errors(homography, view_1, view_2)[0])
    errors = exp14.homography_subset_errors(view_1, view_2, exp14.SUBSETS_6)
    with_inf = errors.copy()
    with_inf[:, :2] = np.inf
    assert exp14.subset_rms_error_score(with_inf, exp14.SUBSETS_6) <= exp14.subset_rms_error_score(errors, exp14.SUBSETS_6)
    assert exp14.sixth_smallest_error_score(with_inf) <= exp14.sixth_smallest_error_score(errors)
    everywhere = np.full_like(errors, np.inf)
    assert exp14.sixth_smallest_error_score(everywhere) == -np.inf == exp14.subset_rms_error_score(everywhere, exp14.SUBSETS_6)
    assert exp14.map_nonfinite(np.array([-np.inf]), "primary")[0] == exp14.LEAST
    assert exp14.sixth_smallest_error_score(everywhere, 0.0) == 0.0


def test_sentinel_collision_and_rank_deficient_dlt_count():
    with pytest.raises(AssertionError):
        exp14.map_nonfinite(np.array([1e299]), "primary")
    view_1 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 3.0], [5.0, 1.0], [3.0, 7.0], [9.0, 2.0], [4.0, 4.5]]) * 50
    homography = np.array([[1.1, 0.1, 5.0], [0.05, 0.9, -3.0], [1e-4, 2e-4, 1.0]])
    mapped = np.column_stack([view_1, np.ones(8)]) @ homography.T
    view_2 = mapped[:, :2] / mapped[:, 2:]
    assert exp14.count_rank_deficient_dlt(view_1, view_2, np.array([[0, 1, 2, 3]])) == 1
    assert exp14.count_rank_deficient_dlt(view_1, view_2, np.array([[0, 3, 4, 5]])) == 0


# ------------------------------------------------------------------ statistics and reproducibility

def test_weighted_auc_matches_sklearn_with_duplicates():
    rng = np.random.default_rng(43)
    scores, labels = np.round(rng.standard_normal(300), 1), rng.integers(0, 2, 300)
    weights = rng.integers(0, 3, (2, 300)).astype(float)
    auc = exp14.weighted_auc(scores, labels, np.vstack([np.ones(300), weights]))
    assert np.isclose(auc[0], roc_auc_score(labels, scores))
    for row, weight in zip(auc[1:], weights):
        repeat = weight.astype(int)
        assert np.isclose(row, roc_auc_score(np.repeat(labels, repeat), np.repeat(scores, repeat)))


def test_threshold_tie_rule_is_strict():
    negatives = np.array([0.0] * 10 + [5.0] * 10)
    threshold = exp14.calibration_threshold(negatives)
    assert threshold == 5.0 and np.mean(negatives > threshold) == 0.0


def test_gric_with_given_models_matches_baseline_scorer():
    view_1, view_2 = noisy(47, 4, 0.5)
    fundamental = cv2.findFundamentalMat(view_1, view_2, cv2.FM_8POINT)[0]
    homography = rsb.fit_homography_dlt(view_1, view_2)
    assert np.isclose(exp14.gric_difference(view_1, view_2, fundamental, homography), rsb.score_gric(view_1, view_2))


def test_stratum_size_estimate():
    assert cop.estimate_stratum_size({5: 100.0, 6: 1.0, 7: 50.0, 8: 80.0}) == 6
    assert cop.estimate_stratum_size({5: 100.0, 6: 100.0, 7: 100.0, 8: 100.0}) == 4


@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="worker processes must inherit the monkeypatched SPLITS; spawned workers re-import "
           "the module and use the real seed bands, so the comparison is not meaningful here",
)
def test_trial_rows_reproduce_alone_and_in_worker_processes(monkeypatch):
    monkeypatch.setattr(exp14, "SPLITS", {"calibration": (95_000_000, 1), "test": (95_500_000, 1)})
    monkeypatch.setattr(exp14, "SAMPLES_PER_K", 1)
    alone = pd.DataFrame(exp14.run_trial(("test", 0, 1)))
    pd.testing.assert_frame_equal(alone, pd.DataFrame(exp14.run_trial(("test", 0, 1))), check_exact=True)
    pooled = exp14.collect_samples(2)
    pd.testing.assert_frame_equal(pooled[pooled["Split"] == "test"].reset_index(drop=True), alone, check_exact=True)
