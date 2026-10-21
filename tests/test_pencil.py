"""
Tests for the pencil cross-ratio module, kappa_J and the max-coverage selector
(docs/MATHEMATICS.md, Sections 2-5). The exact algebraic checks live in
experiments/symbolic/verify_pencil.py; these are fast regression tests.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.consistency.pencil import (  # noqa: E402
    log_abs_pencil_cross_ratios, log_abs_pencil_gradients, pencil_discrepancies, pencil_drift_score, pencil_index,
)
from algorithms.conditioning.scores import compute_conditioning_score, compute_invariant_noise_gain  # noqa: E402
from algorithms.selection.coverage import select_tuples_max_coverage  # noqa: E402
from forensics.geometry.invariants import kappa_j, log_abs_invariants  # noqa: E402


def random_gauge(P, rng):
    A = rng.standard_normal((4, 4))
    scales = rng.uniform(0.3, 3.0, P.shape[1]) * rng.choice([-1, 1], P.shape[1])
    return A @ P * scales


def test_family_size_and_terms():
    quads, terms = pencil_index(12)
    assert len(quads) == 495 and len(terms) == 66 * 210 * 3 == 41580
    assert terms.min() >= 0 and terms.max() < 495


def test_gauge_invariance():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((4, 10))
    assert np.allclose(log_abs_pencil_cross_ratios(P), log_abs_pencil_cross_ratios(random_gauge(P, rng)), atol=1e-8)
    assert np.max(pencil_discrepancies(P, random_gauge(P, rng))) < 1e-8


def test_j_invariants_are_in_the_family():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((4, 6))
    family = log_abs_pencil_cross_ratios(P)
    for value in log_abs_invariants(P):
        assert np.min(np.abs(family - value)) < 1e-9


def test_plucker_relation_is_cross_ratio_identity():
    rng = np.random.default_rng(2)
    P = rng.standard_normal((4, 6))
    B = lambda *q: np.linalg.det(P[:, list(q)])  # noqa: E731
    cr = lambda a, b, c, d, e, f: B(a, b, c, e) * B(a, b, d, f) / (B(a, b, c, f) * B(a, b, d, e))  # noqa: E731
    for a, b in combinations(range(6), 2):
        c, d, e, f = [x for x in range(6) if x not in (a, b)]
        assert cr(a, b, c, d, e, f) + cr(a, b, c, e, d, f) == pytest.approx(1.0, abs=1e-9)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    P = rng.standard_normal((4, 8))
    G = log_abs_pencil_gradients(P)
    h = 1e-6
    for k in (0, 7, 19, 31):
        plus, minus = P.copy(), P.copy()
        plus[k % 4, k // 4] += h
        minus[k % 4, k // 4] -= h
        numeric = (log_abs_pencil_cross_ratios(plus) - log_abs_pencil_cross_ratios(minus)) / (2 * h)
        assert np.allclose(numeric, G[:, k], rtol=1e-4, atol=1e-6)


def test_drift_score_scales_linearly_with_radius():
    """Theorem 4.1: to first order the score is proportional to r for fixed directions."""
    rng = np.random.default_rng(4)
    q = rng.standard_normal((4, 12))
    q /= np.linalg.norm(q, axis=0)
    u, v = rng.standard_normal((2, 4, 12))
    scores = [pencil_drift_score(random_gauge(q + r * u, rng), random_gauge(q + r * v, rng)) for r in (1e-5, 2e-5, 4e-5)]
    assert scores[1] / scores[0] == pytest.approx(2.0, rel=1e-2)
    assert scores[2] / scores[0] == pytest.approx(4.0, rel=1e-2)


def test_drift_score_separates_bounded_from_excessive():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "math_lab"))
    import lab

    rng = np.random.default_rng(5)
    data = [lab.gen_drift(rng) for _ in range(300)]
    y = np.array([label for _, label in data])
    s = np.array([pencil_drift_score(x["A"], x["B"], x["delta"]) for x, _ in data])
    assert lab.auc(y, s) > 0.95


def test_kappa_matches_numerical_definition():
    rng = np.random.default_rng(6)
    for _ in range(5):
        P = rng.standard_normal((4, 6))
        unit = P / np.linalg.norm(P, axis=0)
        assert compute_invariant_noise_gain(P, tuple(range(6))) == pytest.approx(kappa_j(unit), rel=1e-6)


def test_kappa_bounded_by_conditioning_score():
    """Proposition 5.2: kappa_J <= 8 / delta_J <= 2 sqrt(2 L(S))."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        P = rng.standard_normal((4, 6))
        t = tuple(range(6))
        assert compute_invariant_noise_gain(P, t) <= 2 * np.sqrt(2 * compute_conditioning_score(P, t)) * (1 + 1e-9)


def test_max_coverage_selector_covers_and_respects_threshold():
    rng = np.random.default_rng(8)
    P = rng.standard_normal((4, 14))
    threshold = 50.0
    selected, scores = select_tuples_max_coverage(P, compute_invariant_noise_gain, threshold, max_tuples=10, rng=rng)
    assert all(scores[t] < threshold for t in selected)
    coverable = set().union(*scores) if scores else set()
    covered = set().union(*selected) if selected else set()
    assert covered == coverable or len(selected) == 10
    gains = []
    seen = set()
    for t in selected:
        gains.append(len(set(t) - seen))
        seen |= set(t)
    assert all(g > 0 for g in gains) and gains == sorted(gains, reverse=True)


def test_batch_kappa_matches_per_tuple_and_backends():
    from algorithms.conditioning.scores import _batch_invariant_noise_gain_numpy, batch_invariant_noise_gain

    rng = np.random.default_rng(9)
    P = rng.standard_normal((4, 20))
    P[:, 7] = 0.3 * P[:, 0] + P[:, 2] - P[:, 3]           # exact coplanarity: some singular tuples
    T = np.array([sorted(rng.choice(20, 6, replace=False)) for _ in range(500)])
    loop = np.array([compute_invariant_noise_gain(P, tuple(t)) for t in T])
    for batch in (_batch_invariant_noise_gain_numpy(P, T), batch_invariant_noise_gain(P, T)):
        assert np.array_equal(np.isinf(loop), np.isinf(batch))
        finite = np.isfinite(loop)
        assert np.allclose(loop[finite], batch[finite], rtol=1e-9)


def test_sampled_pencils_estimate_the_full_quantile():
    rng = np.random.default_rng(10)
    A = rng.standard_normal((4, 14))
    B = A + 1e-3 * rng.standard_normal((4, 14))
    full = pencil_drift_score(A, B, max_pencils=10 ** 6)
    sampled = pencil_drift_score(A, B, max_pencils=50_000)
    assert sampled == pytest.approx(full, rel=0.05)
    assert pencil_drift_score(A, B) == pencil_drift_score(A, B)          # fixed seed: deterministic


def test_drift_score_ignores_zero_brackets():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((4, 12))
    A[:, 5] = A[:, 0] + A[:, 1] - A[:, 2]                  # brackets containing {0,1,2,5} vanish
    B = A + 1e-4 * rng.standard_normal((4, 12))
    assert np.isfinite(pencil_drift_score(A, B))


def test_point_scores_rank_corrupted_points_first():
    from algorithms.consistency.pencil import pencil_point_scores

    rng = np.random.default_rng(12)
    A = rng.standard_normal((4, 30))
    B = random_gauge(A + 1e-4 * rng.standard_normal((4, 30)), rng)
    B[:, [4, 21]] = rng.standard_normal((4, 2))
    top = set(np.argsort(-pencil_point_scores(A, B))[:2])
    assert top == {4, 21}


def test_video_window_statistic_is_the_pencil_test():
    """Noise-free subset reconstructions of a rigid scene agree in every pencil cross-ratio."""
    from forensics.video.multiview import projective_reconstruction
    from forensics.video.synthetic import normalize_pixels, synthetic_rigid_views
    from algorithms.consistency.pencil import pencil_discrepancies

    rng = np.random.default_rng(13)
    tracks, _ = synthetic_rigid_views(rng, n_points=40, noise_px=0.0)
    views = normalize_pixels(tracks)
    Xa = projective_reconstruction([views[0], views[2], views[4]], 0)["X"]
    Xb = projective_reconstruction([views[1], views[3], views[5]], 0)["X"]
    D = pencil_discrepancies(Xa, Xb)
    assert np.quantile(D[np.isfinite(D)], 0.1) < 1e-6


def test_lemma_10_1_orderings_cover_the_three_pairings_equally():
    """Lemma 10.1: the 24 orderings of a 4-set give each family pairing's |log| value 8 times."""
    from collections import Counter
    from itertools import permutations
    from algorithms.consistency.pencil import PAIRINGS

    rng = np.random.default_rng(14)
    P = rng.standard_normal((4, 6))
    B = lambda *q: np.linalg.det(P[:, list(q)])  # noqa: E731
    value = lambda c, d, e, f: round(abs(np.log(abs(B(0, 1, c, e) * B(0, 1, d, f) / (B(0, 1, c, f) * B(0, 1, d, e))))), 9)  # noqa: E731
    four = (2, 3, 4, 5)
    counts = Counter(value(*order) for order in permutations(four))
    family = {value(four[a1], four[a2], four[b1], four[b2]) for (a1, a2), (b1, b2) in PAIRINGS}
    assert set(counts) == family and set(counts.values()) == {8}
