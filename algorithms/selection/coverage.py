"""
algorithms/selection/coverage.py

Constrained maximum-coverage tuple selection (docs/MATHEMATICS.md, Section 5;
fixes negative finding N4).

Problem: among candidate six-tuples whose conditioning score is below a
threshold (feasible set F), choose at most k tuples maximising the number of
covered points |U_{t in G} t|.

Coverage is monotone and submodular, and the constraint is a cardinality
constraint, so the plain greedy rule "add the feasible tuple with the largest
number of new points" achieves at least (1 - 1/e) of the optimum over F
(Nemhauser, Wolsey and Fisher 1978; cited). The penalised objective of
greedy.py, cov(G) - 0.001 * sum L(t), is not monotone, so that guarantee does
not apply to it, and greedy on it stops after about two tuples (greedy efficiency analysis).

Ties in coverage gain are broken by the smaller conditioning score, which does
not affect the guarantee.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable

import numpy as np


def candidate_tuples(n_points: int, rng: np.random.Generator, max_candidates: int = 10000) -> list[tuple[int, ...]]:
    """All C(n,6) tuples if that is at most max_candidates, otherwise a seeded random sample of distinct tuples."""
    assert n_points >= 6
    if comb(n_points, 6) <= max_candidates:
        return list(combinations(range(n_points), 6))
    # Vectorized rejection sampling: rows of 6 draws, sorted, rows with a repeated
    # point dropped, duplicates removed; each kept row is a uniform 6-subset.
    chosen = np.empty((0, 6), dtype=np.int64)
    while len(chosen) < max_candidates:
        draws = np.sort(rng.integers(0, n_points, size=(2 * (max_candidates - len(chosen)) + 16, 6)), axis=1)
        draws = draws[(draws[:, 1:] != draws[:, :-1]).all(axis=1)]
        merged = np.concatenate([chosen, draws])
        _, first = np.unique(merged, axis=0, return_index=True)
        chosen = merged[np.sort(first)][:max_candidates]          # keep draw order, then truncate
    return [tuple(int(v) for v in row) for row in chosen[np.lexsort(chosen.T[::-1])]]


def select_tuples_max_coverage(
    points_homogeneous: np.ndarray,
    score_fn: Callable[[np.ndarray, tuple], float],
    threshold: float,
    max_tuples: int,
    rng: np.random.Generator | None = None,
    max_candidates: int = 10000,
    batch_score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], float]]:
    """
    Greedy maximum coverage over the feasible candidates {t : score_fn(P, t) < threshold}.

    Returns (selected tuples in selection order, scores of all feasible candidates).
    Stops early when no feasible tuple adds a new point (full coverage of the
    coverable points) or when max_tuples is reached. If batch_score_fn is given
    (e.g. conditioning.scores.batch_invariant_noise_gain) all candidates are
    scored in one call; it must agree with score_fn.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    rng = rng if rng is not None else np.random.default_rng(0)
    candidates = candidate_tuples(points_homogeneous.shape[1], rng, max_candidates)
    if batch_score_fn is not None:
        values = batch_score_fn(points_homogeneous, np.array(candidates, dtype=np.int64))
    else:
        values = [score_fn(points_homogeneous, t) for t in candidates]
    scores: dict[tuple[int, ...], float] = {
        t: float(s) for t, s in zip(candidates, values) if np.isfinite(s) and s < threshold}
    # Vectorized greedy: rows sorted best-conditioned first, so argmax (first maximum)
    # breaks ties towards the smaller score; membership matrix updates the gains.
    remaining = sorted(scores, key=scores.get)
    if not remaining:
        return [], scores
    order = np.array(remaining, dtype=np.int64)
    member = np.zeros((len(order), points_homogeneous.shape[1]), dtype=bool)
    member[np.arange(len(order))[:, None], order] = True
    uncovered = np.ones(points_homogeneous.shape[1], dtype=bool)
    alive = np.ones(len(order), dtype=bool)
    selected: list[tuple[int, ...]] = []
    while len(selected) < max_tuples:
        gains = np.where(alive, member[:, uncovered].sum(axis=1), -1)
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        selected.append(remaining[best])
        alive[best] = False
        uncovered[order[best]] = False
    return selected, scores
