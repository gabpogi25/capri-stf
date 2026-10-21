"""Tests for the frame-coordinate statistic (algorithms/consistency/frame.py; docs/MATHEMATICS.md, items A1, B1)."""

import numpy as np

from algorithms.consistency.frame import choose_frame, frame_discrepancies, frame_log_coordinates


def _random_gauge(rng, P):
    A = rng.normal(size=(4, 4))
    scales = rng.uniform(0.5, 2.0, P.shape[1]) * rng.choice([-1, 1], P.shape[1])
    return (A @ P) * scales


def test_standard_frame_identities():
    rng = np.random.default_rng(1)
    others = rng.normal(size=(4, 4))
    P = np.hstack([np.eye(4), np.ones((4, 1)), others])
    coords = frame_log_coordinates(P, (0, 1, 2, 3, 4)).reshape(-1, 3)
    x, y, z, w = others
    expected = np.log(np.abs(np.stack([z / w, y / w, x / w], 1)))
    assert np.allclose(coords, expected, atol=1e-12)
    wrong = np.log(np.abs(np.stack([y / w, z / w, x / w], 1)))      # control: wrong order must differ
    assert not np.allclose(coords, wrong, atol=1e-6)


def test_gauge_invariance_and_zero_discrepancy():
    rng = np.random.default_rng(2)
    P = rng.normal(size=(4, 12))
    Q = _random_gauge(rng, P)
    frame = choose_frame(P, Q)
    assert np.allclose(frame_log_coordinates(P, frame), frame_log_coordinates(Q, frame), atol=1e-9)
    assert np.max(frame_discrepancies(P, Q, frame)) < 1e-9
    R = Q.copy()
    R[:, 7] += 0.05 * rng.normal(size=4)                              # move one point: must be detected
    assert np.max(frame_discrepancies(P, R, frame)) > 1e-3


def test_choose_frame_is_symmetric_and_deterministic():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(4, 9))
    Q = _random_gauge(rng, P)
    f1, f2 = choose_frame(P, Q), choose_frame(Q, P)
    assert f1 == f2 == choose_frame(P, Q)
    assert len(set(f1)) == 5


def test_counts():
    rng = np.random.default_rng(4)
    for n in (6, 10, 30):
        P = rng.normal(size=(4, n))
        assert frame_log_coordinates(P, (0, 1, 2, 3, 4)).shape == (3 * (n - 5),)


def test_cross4_matches_determinant():
    from algorithms.consistency.frame import _cross4
    rng = np.random.default_rng(5)
    M = rng.normal(size=(4, 3))
    x = rng.normal(size=4)
    assert np.isclose(_cross4(M) @ x, np.linalg.det(np.column_stack([M, x])))


def test_choose_frame_maximises_the_margin_exhaustively():
    from itertools import combinations
    rng = np.random.default_rng(6)
    P, Q = rng.normal(size=(4, 8)), rng.normal(size=(4, 8))

    def margin(f):
        out = np.inf
        for X in (P, Q):
            U = X / np.linalg.norm(X, axis=0)
            quads = list(combinations(f, 4)) + [tuple(f[t] for t in trio) + (k,) for k in range(8) if k not in f
                                                for trio in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
            out = min(out, min(abs(np.linalg.det(U[:, list(q)])) for q in quads))
        return out
    best = max(combinations(range(8), 5), key=margin)
    assert choose_frame(P, Q) == best
