"""
algorithms/syzygy/residual.py

Plucker consistency residuals and drift-admissibility bounds for six-point
tuples in P^3.

Background
----------
The 15 four-point brackets of six points in P^3 satisfy 15 three-term
Grassmann-Plucker relations (one per fixed pair {a, b}):

    [abcd][abef] - [abce][abdf] + [abcf][abde] = 0.

Three of them are used as Sigma_1, Sigma_2, Sigma_3 and define the
residual rho(S) = max |Sigma_k|.

Two facts limit what rho(S) can detect:

* docs/MATHEMATICS.md, Corollary 2.6: if the 15 brackets are recomputed from ONE set
  of six points (any points, however corrupted), every Plucker relation holds
  identically, so rho(S) is zero up to rounding. rho(S) can only be non-zero
  when the brackets come from different measurements (independent estimation,
  see synthetic_data.generator.estimate_brackets_independently).
* Proposition 6.2: Sigma_1..Sigma_3 constrain only 3 of the 6 locally independent
  directions. The point b* = e_1234 + e_1456 has Sigma_1 = Sigma_2 = Sigma_3 = 0
  but violates the {1,4} relation. compute_full_plucker_residual() uses all 15.

Bounds
------
The first-order bound is tau = 24 * delta. docs/MATHEMATICS.md, Theorem 6.1
gives the exact bound used here:

    eps = (1 + delta)^4 - 1,   tau_exact(delta) = 3 * eps * (2 + eps)
                                                = 24 delta + O(delta^2).

Under the model in which every point of a bracket is replaced by a vector
within distance delta of its unit representative, rho(S) <= tau_exact(delta).
"""

from __future__ import annotations
from itertools import combinations
import numpy as np
from algorithms.invariants.brackets import get_bracket


BracketTable = dict[tuple[int, int, int, int], float]


def _sorted_point_indices(brackets: BracketTable) -> list[int]:
    point_indices = sorted({index for quad in brackets for index in quad})
    if len(point_indices) != 6:
        raise ValueError(
            f"Expected brackets over exactly 6 points, got {len(point_indices)}: {point_indices}"
        )
    return point_indices


def _plucker_relation_terms(
    brackets: BracketTable, a: int, b: int, c: int, d: int, e: int, f: int
) -> tuple[float, float, float]:
    """The three signed terms of [abcd][abef] - [abce][abdf] + [abcf][abde]."""
    term_1 = get_bracket(brackets, a, b, c, d) * get_bracket(brackets, a, b, e, f)
    term_2 = get_bracket(brackets, a, b, c, e) * get_bracket(brackets, a, b, d, f)
    term_3 = get_bracket(brackets, a, b, c, f) * get_bracket(brackets, a, b, d, e)
    return term_1, -term_2, term_3


def compute_syzygy_residual(brackets: BracketTable) -> float:
    """
    Residual rho(S) = max(|Sigma_1|, |Sigma_2|, |Sigma_3|).

    Sigma_1 = [1234][1256] - [1235][1246] + [1236][1245]
    Sigma_2 = [1345][2346] - [1346][2345] - [1234][3456]
    Sigma_3 = [1256][3456] - [1356][2456] + [1456][2356]

    Args:
        brackets: the 15 brackets of one six-point tuple. For the bound
                  tau_exact to apply, each bracket must use representatives
                  within delta of the TRUE unit representatives; measured
                  vectors need not be unit, and renormalising them can move
                  them slightly beyond delta (docs/MATHEMATICS.md, Thm 6.1).

    Returns:
        rho(S) >= 0.
    """
    t = _sorted_point_indices(brackets)

    def B(i: int, j: int, k: int, l: int) -> float:
        return get_bracket(brackets, t[i], t[j], t[k], t[l])

    sigma_1 = B(0, 1, 2, 3) * B(0, 1, 4, 5) - B(0, 1, 2, 4) * B(0, 1, 3, 5) + B(0, 1, 2, 5) * B(0, 1, 3, 4)
    sigma_2 = B(0, 2, 3, 4) * B(1, 2, 3, 5) - B(0, 2, 3, 5) * B(1, 2, 3, 4) - B(0, 1, 2, 3) * B(2, 3, 4, 5)
    sigma_3 = B(0, 1, 4, 5) * B(2, 3, 4, 5) - B(0, 2, 4, 5) * B(1, 3, 4, 5) + B(0, 3, 4, 5) * B(1, 2, 4, 5)
    return float(max(abs(sigma_1), abs(sigma_2), abs(sigma_3)))


def compute_all_plucker_residuals(brackets: BracketTable) -> list[float]:
    """Signed value of each of the 15 Grassmann-Plucker relations (one per fixed pair)."""
    t = _sorted_point_indices(brackets)
    residuals = []
    for pair in combinations(range(6), 2):
        free = [i for i in range(6) if i not in pair]
        a, b = (t[i] for i in pair)
        c, d, e, f = (t[i] for i in free)
        residuals.append(float(sum(_plucker_relation_terms(brackets, a, b, c, d, e, f))))
    return residuals


def compute_full_plucker_residual(brackets: BracketTable) -> float:
    """max |relation| over all 15 Plucker relations (not blind at b*, Proposition 6.2)."""
    return float(max(abs(value) for value in compute_all_plucker_residuals(brackets)))


def compute_relative_plucker_residual(brackets: BracketTable) -> float:
    """
    Scale-free residual rho_rel = max_k |R_k| / sum_terms |B_i B_j| over all 15
    relations.

    The absolute residual shrinks quadratically as brackets shrink, so a tuple
    near a degeneracy looks "more consistent" than a generic one. Dividing each
    relation by the sum of its absolute terms removes that artifact. Returns 0
    when every term of every relation vanishes.
    """
    t = _sorted_point_indices(brackets)
    worst = 0.0
    for pair in combinations(range(6), 2):
        free = [i for i in range(6) if i not in pair]
        a, b = (t[i] for i in pair)
        c, d, e, f = (t[i] for i in free)
        terms = _plucker_relation_terms(brackets, a, b, c, d, e, f)
        scale = sum(abs(term) for term in terms)
        if scale > 0.0:
            worst = max(worst, abs(sum(terms)) / scale)
    return float(worst)


def compute_first_order_bound(delta_drift: float) -> float:
    """First-order bound tau = 24 * delta (for comparison with the exact bound)."""
    assert delta_drift >= 0.0, "delta_drift must be non-negative."
    return 24.0 * float(delta_drift)


def compute_exact_admissibility_bound(delta_drift: float) -> float:
    """Theorem 6.1: tau_exact(delta) = 3 eps (2 + eps), eps = (1 + delta)^4 - 1."""
    assert delta_drift >= 0.0, "delta_drift must be non-negative."
    eps = (1.0 + float(delta_drift)) ** 4 - 1.0
    return 3.0 * eps * (2.0 + eps)


def compute_drift_threshold(delta_drift: float) -> float:
    """
    Drift threshold tau(S) used by the pipeline: the exact bound of Theorem 6.1.

    It depends only on the deformation bound delta, not on L(S).
    """
    return compute_exact_admissibility_bound(delta_drift)


def is_tuple_drift_admissible(brackets: BracketTable, delta_drift: float) -> bool:
    """True if rho(S) <= tau_exact(delta_drift)."""
    return compute_syzygy_residual(brackets) <= compute_exact_admissibility_bound(delta_drift)


def compute_consistency_residual(
    invariants_a: tuple[float, float, float],
    invariants_b: tuple[float, float, float],
) -> float:
    """
    Multi-view invariant agreement max_k |J_k^A - J_k^B| for the same six
    physical points measured in two independent ways (README definition).
    """
    assert len(invariants_a) == 3 and len(invariants_b) == 3
    return float(max(abs(x - y) for x, y in zip(invariants_a, invariants_b)))


def compute_log_invariant_discrepancy(
    invariants_a: tuple[float, float, float],
    invariants_b: tuple[float, float, float],
) -> np.ndarray:
    """Delta_J = log|J^A| - log|J^B| per invariant."""
    a = np.abs(np.asarray(invariants_a, dtype=float))
    b = np.abs(np.asarray(invariants_b, dtype=float))
    return np.log(a) - np.log(b)
