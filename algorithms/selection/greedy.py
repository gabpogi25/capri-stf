"""
algorithms/selection/greedy.py

Greedy tuple selection for six-point invariant evaluation.

Because the number of six-point tuples grows as C(n,6), exhaustive evaluation
is infeasible for large n. This module selects a small set G of
well-conditioned tuples.

Objective f(G)
--------------
Let V = {0, ..., n-1} be the point indices, T the candidate tuples that pass
the stratum filter and the conditioning gate L(t) < threshold, and for a set
of selected tuples G let cov(G) = | union of t over t in G | (points covered).
The selection maximizes

    f(G) = w_cov * cov(G) - w_pen * sum_{t in G} L(t)

subject to |G| <= k. The coverage part cov(G) is monotone submodular, so plain
greedy on cov alone has the (1 - 1/e) guarantee of Nemhauser et al. (1978).
The penalty term makes f non-monotone (adding a tuple can lower f), so that
guarantee does NOT carry over to f; the greedy efficiency analysis measure greedy
against the exact optimum only where it is computable (n = 10) and against the
trivial upper bound cov = n elsewhere.

"Coverage" therefore means: fraction of the n points that appear in at least
one selected, well-conditioned tuple. Greedy stops early once every point is
covered.

Implementation: Minoux's lazy greedy with a max-heap. The marginal gain
g(t | G) = w_cov * |t minus covered(G)| - w_pen * L(t) is non-increasing in G,
so stale heap scores are valid upper bounds.

Backends: the Python path enumerates or samples candidates with the given rng
and honours allowed_strata; the C++ path (cpp_core/src/greedy.cpp) samples
candidates with its own generator seeded from rng and applies only the
conditioning gate. The two paths can therefore select different tuples for
the same seed; every experiment records which backend it used.

Reference: Nemhauser et al. (1978), Minoux (1978).
"""

from __future__ import annotations
import heapq
import numpy as np
from algorithms.backend import get_cpp_module
from itertools import combinations
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.degeneracy.stratification import (
    classify_degeneracy_stratum,
    DegeneracyStratum,
)


def enumerate_six_point_tuples(n_points: int) -> list[tuple[int, int, int, int, int, int]]:
    """
    Enumerate all C(n,6) six-point index tuples for n points.

    Args:
        n_points: Total number of candidate keypoints.

    Returns:
        List of all 6-tuples of distinct indices in range(n_points).
    """
    assert n_points >= 6, f"Need at least 6 points, got {n_points}."
    return list(combinations(range(n_points), 6))


def _compute_greedy_score(
    tuple_indices: tuple,
    covered_points: set[int],
    conditioning_score: float,
    coverage_weight: float,
    conditioning_penalty_weight: float,
) -> float:
    """
    Compute the greedy marginal gain for a candidate tuple.

    score = coverage_weight * (new points covered) - penalty * L(S)

    The coverage term is non-increasing in |covered_points| (submodular).
    The penalty term is constant per tuple. Therefore the total score is
    non-increasing, making lazy greedy valid.
    """
    new_coverage = len(set(tuple_indices) - covered_points)
    return coverage_weight * new_coverage - conditioning_penalty_weight * conditioning_score


def select_tuples_greedy(
    points_homogeneous: np.ndarray,
    conditioning_threshold: float = 1000.0,
    max_tuples: int = 500,
    coverage_weight: float = 1.0,
    conditioning_penalty_weight: float = 0.001,
    allowed_strata: set[DegeneracyStratum] = None,
    rng: np.random.Generator = None,
    candidate_sample_size: int = 10000
) -> list[tuple[int, int, int, int, int, int]]:
    """
    Select a subset of well-conditioned six-point tuples using lazy greedy.

    Uses Minoux's lazy greedy algorithm: a max-heap stores upper bounds
    on marginal gains. At each step, the top element is popped and its
    score recomputed against the current covered set. If still maximal,
    it is selected; otherwise it is reinserted with the updated score.

    This achieves the same (1-1/e) approximation guarantee as standard
    greedy but with amortized O(K log K) runtime instead of O(K * T).

    Args:
        points_homogeneous: (4, N) array of homogeneous coordinates.
        conditioning_threshold: Maximum acceptable L(S).
        max_tuples: Maximum number of tuples to select.
        coverage_weight: Weight on new coverage in greedy score.
        conditioning_penalty_weight: Weight on L(S) penalty in greedy score.
        allowed_strata: Accepted degeneracy strata. Defaults to {GENERIC, BOUNDARY}.
        rng: Numpy random generator for reproducible candidate sampling.
        candidate_sample_size: Number of tuples to sample before greedy scoring.
                               Set to None to enumerate all C(n,6) (expensive).

    Returns:
        List of selected 6-tuples.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    n_points = points_homogeneous.shape[1]
    assert n_points >= 6, f"Need at least 6 points, got {n_points}."

    capri_cpp = get_cpp_module()
    if capri_cpp is not None:
        import math
        seed_val = 0 if rng is None else int(rng.integers(0, 2147483647))
        target_sample_size = candidate_sample_size if candidate_sample_size is not None else 10000
        max_possible = math.comb(n_points, 6)
        if target_sample_size > max_possible:
            target_sample_size = max_possible

        return capri_cpp.select_tuples_greedy(
            points=points_homogeneous,
            max_tuples=max_tuples,
            conditioning_threshold=conditioning_threshold,
            coverage_weight=coverage_weight,
            penalty_weight=conditioning_penalty_weight,
            candidate_sample_size=target_sample_size,
            seed=seed_val
        )

    if allowed_strata is None:
        allowed_strata = {DegeneracyStratum.GENERIC, DegeneracyStratum.BOUNDARY}

    if rng is None:
        rng = np.random.default_rng(seed=0)

    # Generate candidate tuples
    if n_points > 25:
        # Avoid combinatorial explosion by sampling directly
        candidate_set = set()
        num_samples = candidate_sample_size if candidate_sample_size is not None else 10000
        while len(candidate_set) < num_samples:
            t = tuple(np.sort(rng.choice(n_points, size=6, replace=False)))
            candidate_set.add(t)
        candidates = list(candidate_set)
    else:
        all_tuples = enumerate_six_point_tuples(n_points)
        if candidate_sample_size is not None and len(all_tuples) > candidate_sample_size:
            indices = rng.choice(len(all_tuples), size=candidate_sample_size, replace=False)
            candidates = [all_tuples[i] for i in indices]
        else:
            candidates = all_tuples

    # Pre-filter by degeneracy stratum and conditioning score
    viable: list[tuple] = []
    conditioning_scores: list[float] = []
    for t in candidates:
        stratum = classify_degeneracy_stratum(points_homogeneous, t)
        if stratum not in allowed_strata:
            continue
        score_l = compute_conditioning_score(points_homogeneous, t)
        if score_l >= conditioning_threshold:
            continue
        viable.append(t)
        conditioning_scores.append(score_l)

    if not viable:
        return []

    # Lazy greedy selection (Minoux 1978)
    # Initialize max-heap with upper-bound scores.
    # Python heapq is a min-heap, so negate scores for max-heap behavior.
    covered_points: set[int] = set()
    heap: list[tuple[float, int]] = []
    for k, (t, l_s) in enumerate(zip(viable, conditioning_scores)):
        score = _compute_greedy_score(
            t, covered_points, l_s, coverage_weight, conditioning_penalty_weight
        )
        heapq.heappush(heap, (-score, k))

    selected: list[tuple] = []

    for _ in range(max_tuples):
        if not heap:
            break
        if len(covered_points) == n_points:
            break

        # Lazy evaluation loop: pop, recompute, check if still best
        while heap:
            neg_stale_score, k = heapq.heappop(heap)

            current_score = _compute_greedy_score(
                viable[k], covered_points, conditioning_scores[k],
                coverage_weight, conditioning_penalty_weight,
            )

            if not heap:
                # Last element, must be best
                if current_score > 0 or len(selected) == 0:
                    selected.append(viable[k])
                    covered_points.update(viable[k])
                break

            next_neg_score = heap[0][0]
            if -current_score <= next_neg_score:
                # Recomputed score is still >= all remaining upper bounds
                # (remember: negated, so -current <= next_neg means
                #  current >= -next_neg = upper bound of next)
                if current_score > 0 or len(selected) == 0:
                    selected.append(viable[k])
                    covered_points.update(viable[k])
                break
            else:
                # Score dropped below the next candidate's upper bound;
                # reinsert with updated score and try next
                heapq.heappush(heap, (-current_score, k))

    return selected
