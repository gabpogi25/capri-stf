"""Checks of algorithms/consistency/variants.py and forensics/stats/combine.py (evaluation_plan arms)."""

import numpy as np

from algorithms.consistency.pencil import pencil_index
from algorithms.consistency.variants import (asinh_discrepancies, gauge_fit, log_discrepancies,
                                             pencil_gradient_norms, sampled_pencils)
from forensics.stats.combine import bonferroni_min, cauchy_combination, e_process_alarm, e_product


def _gauge(points, rng):
    A = rng.standard_normal((4, 4)) + 2 * np.eye(4)
    scales = rng.uniform(0.5, 2.0, points.shape[1]) * rng.choice([-1, 1], points.shape[1])
    return (A @ points) * scales


def test_additive_gauge_identity_exact():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.uniform(-1, 1, (3, 10)), np.ones((1, 10))])
    quads, _ = pencil_index(10)
    fit = gauge_fit(X, _gauge(X, rng), quads)
    assert fit["statistic"] < 1e-9


def test_gauge_fit_responds_and_localises():
    rng = np.random.default_rng(2)
    X = np.vstack([rng.uniform(-1, 1, (3, 12)), np.ones((1, 12))])
    Y = _gauge(X, rng)
    Y[:3, 5] += 0.3 * Y[3, 5]
    quads, _ = pencil_index(12)
    fit = gauge_fit(X, Y, quads)
    assert fit["statistic"] > 1e-3
    assert int(np.argmax(fit["point_scores"])) == 5


def test_signed_cross_ratio_invariant_and_asinh_zero_on_gauge():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.uniform(-1, 1, (3, 9)), np.ones((1, 9))])
    pencils = sampled_pencils(9, 500, 0)
    Y = _gauge(X, rng)
    assert np.nanmax(asinh_discrepancies(X, Y, pencils)) < 1e-9
    assert np.nanmax(log_discrepancies(X, Y, pencils)) < 1e-9


def test_gradient_norm_matches_finite_differences():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((4, 7))
    X /= np.linalg.norm(X, axis=0)
    pencils = sampled_pencils(7, 5, 1)

    def logcr(P):
        b = np.linalg.det(np.transpose(P[:, pencils.reshape(-1, 4)], (1, 0, 2))).reshape(-1, 4)
        return np.log(np.abs(b[:, 0] * b[:, 1] / (b[:, 2] * b[:, 3])))

    h, grad = 1e-6, np.zeros((5, 28))
    for k in range(28):
        E = np.zeros(28); E[k] = h
        grad[:, k] = (logcr(X + E.reshape(7, 4).T) - logcr(X - E.reshape(7, 4).T)) / (2 * h)
    assert np.allclose(np.linalg.norm(grad, axis=1), pencil_gradient_norms(X, pencils), rtol=1e-5)


def test_combiners_validity_and_alarm():
    rng = np.random.default_rng(5)
    p_null = rng.uniform(size=(4000, 10))
    assert abs(np.mean([cauchy_combination(p) <= 0.05 for p in p_null]) - 0.05) < 0.015
    assert np.mean([bonferroni_min(p) <= 0.05 for p in p_null]) <= 0.06
    assert np.mean([e_product(p) >= 20 for p in p_null]) <= 0.05
    assert e_process_alarm(np.full(20, 1e-3), 0.05) == 1      # e = 0.5 * 1000 ** 0.5 = 15.8 < 20, then 250
    assert e_process_alarm(np.full(20, 0.9), 0.05) is None


def test_fast_ba_jacobian_and_convergence():
    from forensics.video.fast_ba import _jacobian, _residuals, bundle_adjust_lm
    rng = np.random.default_rng(6)
    X = np.vstack([rng.uniform(-1, 1, (3, 15)), np.ones((1, 15))])
    cams = np.stack([np.hstack([np.eye(3), np.zeros((3, 1))])] +
                    [np.hstack([np.eye(3) + 0.1 * rng.standard_normal((3, 3)), rng.standard_normal((3, 1)) + [[0], [0], [4]]])
                     for _ in range(2)])
    X[2] += 4
    obs = np.stack([(c @ X)[:2].T / (c @ X)[2][:, None] for c in cams])
    J = _jacobian(cams, X)
    h, k = 1e-7, 20
    e = np.zeros(J.shape[1]); e[k] = h
    def unpack(p):
        c = cams.copy(); c[1:] += p[:24].reshape(2, 3, 4); return c, X + p[24:].reshape(15, 4).T
    num = (_residuals(*unpack(e), obs) - _residuals(*unpack(-e), obs)) / (2 * h)
    assert np.allclose(num, J[:, k], atol=1e-6)
    noisy = cams.copy(); noisy[1:] += 0.01 * rng.standard_normal((2, 3, 4))
    _, _, cost = bundle_adjust_lm(list(noisy), X + 0.01 * rng.standard_normal(X.shape), list(obs))
    assert cost < 1e-16


def test_tyler_sigma6_is_frame_and_representative_free():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "eval"))
    from coplanarity_robustness import tyler_sigma6
    rng = np.random.default_rng(7)
    X = np.vstack([rng.uniform(-1, 1, (3, 8)), np.ones((1, 8))])
    assert abs(tyler_sigma6(X) - tyler_sigma6(_gauge(X, rng))) < 1e-6
