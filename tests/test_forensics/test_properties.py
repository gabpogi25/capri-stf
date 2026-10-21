"""
tests/forensics/test_properties.py

Property tests T01-T10 of the forensic code. Each test can fail
for the reason it names.
"""

import hashlib
import json
import random
from itertools import combinations

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize

from algorithms.degeneracy.stratification import largest_coplanar_subset_size
from forensics.config import load_config
from forensics.geometry.brackets import (
    b_star_table, exact_bracket_table, fifteen_plucker_relations, manuscript_sigmas,
)
from forensics.geometry.invariants import (
    amplification_bound, invariants, j1_delta_min, log_abs_invariants, tau_exact,
)
from forensics.pipeline_image import analyze_gray
from forensics.structure.collinearity import orthogonal_regression_angle
from forensics.structure.lines import orientation_sigma
from forensics.structure.synthetic import render_perspective_scene, synthetic_segments
from forensics.structure.vp_crossfit import cross_fitted_vp_test


def _unit(P):
    return P / np.linalg.norm(P, axis=0, keepdims=True)


def _float_table(P):
    return {q: float(np.linalg.det(P[:, list(q)])) for q in combinations(range(6), 4)}


def test_t01_plucker_relations_vanish_exactly_on_integer_configurations():
    rng = random.Random(1)
    for _ in range(200):
        points = [[rng.randint(-20, 20) for _ in range(4)] for _ in range(6)]
        assert all(v == 0 for v in fifteen_plucker_relations(exact_bracket_table(points)))


def test_t02_rho_on_coherent_perturbations_is_rounding_only():
    rng = np.random.default_rng(2)
    for scale in (1e-6, 1e-2, 1.0, 10.0):
        for _ in range(100):
            P = _unit(rng.standard_normal((4, 6)) + scale * rng.standard_normal((4, 6)))
            sigmas = manuscript_sigmas(_float_table(P))
            assert max(abs(s) for s in sigmas) <= 1e-12


def test_t03_invariants_unchanged_under_gl4():
    rng = np.random.default_rng(3)
    errors = []
    for _ in range(500):
        P = rng.standard_normal((4, 6))
        A = rng.standard_normal((4, 4))
        if np.linalg.cond(A) > 1e6:
            continue
        before, after = invariants(P), invariants(A @ P)
        errors.append(np.max(np.abs(after - before) / np.maximum(np.abs(before), 1e-300)))
    assert np.median(errors) <= 1e-12


def test_t04_b_star_counterexample():
    table = b_star_table()
    assert manuscript_sigmas(table) == (0, 0, 0)
    relations = fifteen_plucker_relations(table)
    nonzero = [(pair, value) for pair, value in zip(combinations(range(6), 2), relations) if value != 0]
    assert nonzero == [((0, 3), 1)]          # the {1, 4} relation (1-based) equals 1


def test_t05_amplification_bound_holds():
    rng = np.random.default_rng(5)
    violations = checked = 0
    for _ in range(4000):
        P = _unit(rng.standard_normal((4, 6)))
        delta_min = j1_delta_min(P)
        sigma = 10.0 ** rng.uniform(-6, -2)
        bound = amplification_bound(sigma, delta_min)
        if not np.isfinite(bound):
            continue
        E = rng.standard_normal((4, 6))
        E *= sigma * rng.uniform(0, 1, 6) / np.linalg.norm(E, axis=0)
        change = abs(log_abs_invariants(P + E)[0] - log_abs_invariants(P)[0])
        violations += change > bound
        checked += 1
    assert checked > 1000 and violations == 0


def _rho_under_perturbation(P, delta, directions):
    table_true = _float_table(P)
    perturbed = {}
    for k, q in enumerate(combinations(range(6), 4)):
        cols = P[:, list(q)] + delta * directions[k]
        perturbed[q] = float(np.linalg.det(cols))
    del table_true
    return max(abs(s) for s in manuscript_sigmas(perturbed))


def test_t06_exact_admissibility_bound_random_and_adversarial():
    rng = np.random.default_rng(6)
    for delta in (1e-3, 1e-2, 1e-1):
        for _ in range(500):
            P = _unit(rng.standard_normal((4, 6)))
            directions = rng.standard_normal((15, 4, 4))
            directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
            directions *= rng.uniform(0, 1, (15, 1, 4))
            assert _rho_under_perturbation(P, delta, directions) <= tau_exact(delta)
    # Adversarial search: maximize rho over bracket-wise perturbations of norm <= delta.
    delta = 0.05
    P = _unit(rng.standard_normal((4, 6)))

    def negative_rho(x):
        d = x.reshape(15, 4, 4)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1.0)
        return -_rho_under_perturbation(P, delta, d)

    best = minimize(negative_rho, rng.standard_normal(15 * 16), method="Powell", options={"maxiter": 3000})
    assert -best.fun <= tau_exact(delta)


def test_t07_rank_classifier_has_no_false_negatives_under_bounded_noise():
    rng = np.random.default_rng(7)
    for m in (4, 5, 6):
        for _ in range(200):
            pts = rng.uniform(-1, 1, (6, 3))
            normal = rng.standard_normal(3)
            normal /= np.linalg.norm(normal)
            pts[:m] -= np.outer(pts[:m] @ normal - pts[0] @ normal, normal)
            P = _unit(np.vstack([np.ones(6), pts.T]))
            sigma = 1e-4
            E = rng.standard_normal((4, 6))
            E *= sigma / np.linalg.norm(E, axis=0)
            assert largest_coplanar_subset_size(P + E, tuple(range(6)), tolerance=np.sqrt(6) * sigma) >= m


@pytest.mark.slow
def test_t08_cross_fitted_vp_z_scores_are_standard_normal():
    config = load_config(overrides={"vanishing_points": {"ransac_iters": 300}})
    rng = np.random.default_rng(8)
    z = []
    while len(z) < 20000:
        z += [r["z"] for r in cross_fitted_vp_test(synthetic_segments(rng), config, rng)]
    assert stats.kstest(np.asarray(z), "norm").statistic <= 0.02


def test_t09_segment_orientation_variance_matches_model():
    rng = np.random.default_rng(9)
    sigma_p, l_c = 0.7, 2.0
    for length in (80.0, 200.0):
        angles = []
        for _ in range(10000):
            n_groups = int(length / l_c)
            noise = np.repeat(rng.normal(0, sigma_p, n_groups), int(l_c))
            t = np.arange(len(noise), dtype=float)
            angles.append(orthogonal_regression_angle(np.column_stack([t, noise])))
        angles = (np.asarray(angles) + np.pi / 2) % np.pi - np.pi / 2
        predicted = orientation_sigma(np.array([length]), sigma_p, l_c)[0] ** 2
        assert abs(np.var(angles) / predicted - 1.0) <= 0.10


def test_t10_same_seed_and_config_give_identical_output():
    config = load_config(overrides={"vanishing_points": {"ransac_iters": 200}})
    gray = render_perspective_scene(np.random.default_rng(10))

    def digest():
        evidence = analyze_gray(gray, config, np.random.default_rng(123))
        payload = json.dumps(evidence.summary(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    assert digest() == digest()
