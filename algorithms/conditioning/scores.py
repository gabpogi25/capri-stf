"""
algorithms/conditioning/scores.py

Numerical conditioning scores for six-point projective tuples.

The conditioning score L(S) quantifies how close a six-point tuple is to
bracket degeneracy, using unit-sphere-normalized bracket volumes.

After normalizing all homogeneous representatives to the unit sphere (||p_i||=1),
the maximum norm M = 1, so C_br = 4 * M^3 = 4.0 is a universal constant.
The conditioning score simplifies to:

    L(S) = 8.0 / delta_min(S)^2

Invariance: L(S) is invariant
under per-point rescaling p_i -> lambda_i p_i, because normalization absorbs
the scale factor. It is NOT invariant under general projective maps
A in PGL(4): with p = (e1, e2, e3, e4) and A sending e2 to e1 + eps*e2 (fixing
the others), the normalized bracket falls from 1 to eps / sqrt(1 + eps^2), so
L(S) rises from 8 to 8 (1 + eps^2) / eps^2. L(S) therefore depends on the
coordinate chart in which points are expressed. It is used as a conditioning
proxy in a fixed chart, not as a projective quantity. Its precise meaning
(docs/MATHEMATICS.md, Proposition 5.2): sqrt(L(S)/8) = 1/delta_min bounds the
noise amplification of log J from above. compute_singular_value_conditioning()
below is a second scale-free quantity (also chart-dependent).

Reference: Hartley & Zisserman (2003), docs/MATHEMATICS.md.
"""

from __future__ import annotations
import numpy as np
from itertools import combinations
from algorithms.invariants.brackets import compute_bracket
from algorithms.backend import get_cpp_module
from algorithms.utils import normalize_to_unit_sphere

# After unit-sphere normalization, M = 1 always, so C_br = 4 * 1^3 = 4.0.
_C_BR_NORMALIZED: float = 4.0


def compute_delta_min(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int]
) -> float:
    """
    Compute delta_min(S) = min over all 4-subsets T of S of |[p_T]|.

    Points are normalized to the unit sphere before bracket computation,
    making delta_min a true geometric degeneracy measure independent of
    coordinate scaling.

    Args:
        points_homogeneous: (4, N) array.
        tuple_indices: 6-tuple of column indices.

    Returns:
        delta_min (float), the minimum normalized bracket magnitude.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6

    points_normalized = normalize_to_unit_sphere(points_homogeneous)

    min_bracket = float('inf')
    for quad in combinations(tuple_indices, 4):
        bracket_value = abs(compute_bracket(points_normalized, quad, pre_normalized=True))
        if bracket_value < min_bracket:
            min_bracket = bracket_value

    return float(min_bracket)


def compute_conditioning_score(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int]
) -> float:
    """
    Compute the conditioning score L(S) = 2 * C_br / delta_min(S)^2.

    With unit-sphere normalization, C_br = 4.0, so L(S) = 8.0 / delta_min^2.
    The score does not depend on the scale of each homogeneous representative,
    but it does depend on the projective chart (see module docstring).

    Args:
        points_homogeneous: (4, N) array.
        tuple_indices: 6-tuple of column indices.

    Returns:
        L(S) (float). Returns float('inf') if delta_min is effectively zero.
    """
    capri_cpp = get_cpp_module()
    if capri_cpp is not None:
        return capri_cpp.compute_conditioning_score(points_homogeneous, tuple_indices)

    delta_min = compute_delta_min(points_homogeneous, tuple_indices)
    if delta_min < 1e-15:
        return float('inf')

    return float(2.0 * _C_BR_NORMALIZED / (delta_min ** 2))


def batch_conditioning_scores(
    points_homogeneous: np.ndarray,
    all_tuple_indices: list[tuple[int, int, int, int, int, int]]
) -> np.ndarray:
    """
    Compute L(S) for a list of tuples and return as a numpy array.

    Args:
        points_homogeneous: (4, N) array.
        all_tuple_indices: List of 6-tuples.

    Returns:
        (K,) array of conditioning scores, one per tuple.
    """
    capri_cpp = get_cpp_module()
    if capri_cpp is not None:
        return capri_cpp.compute_conditioning_scores_batched(points_homogeneous, all_tuple_indices)

    scores = np.array([
        compute_conditioning_score(points_homogeneous, t)
        for t in all_tuple_indices
    ])
    return scores


def is_tuple_well_conditioned(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    conditioning_threshold: float = 1000.0
) -> bool:
    """
    Return True if the tuple's conditioning score is below the threshold.

    Args:
        points_homogeneous: (4, N) array.
        tuple_indices: 6-tuple of column indices.
        conditioning_threshold: Maximum acceptable L(S). Default 1000.0
                                corresponds to delta_min ~ 0.089 on the
                                unit sphere.

    Returns:
        Boolean indicating whether the tuple is well-conditioned.
    """
    score = compute_conditioning_score(points_homogeneous, tuple_indices)
    return score < conditioning_threshold

def compute_singular_value_conditioning(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int]
) -> float:
    """
    Scale-free conditioning sigma_4 / sigma_1 of the 4x6 matrix of
    unit-normalized representatives.

    sigma_4 is the Frobenius distance of the six points to the nearest
    rank-3 (all-coplanar) configuration (Eckart-Young). Values near 0 mean
    the whole tuple is close to one plane. Unlike delta_min it does not look
    at individual quadruples, so it complements L(S) rather than replacing it.

    Returns:
        sigma_4 / sigma_1 in [0, 1].
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6
    points_normalized = normalize_to_unit_sphere(points_homogeneous[:, list(tuple_indices)])
    singular_values = np.linalg.svd(points_normalized, compute_uv=False)
    return float(singular_values[3] / singular_values[0])


# A bracket of unit vectors below this is numerically zero: kappa_J would exceed
# 1e12, far outside the first-order regime (docs/MATHEMATICS.md, Prop. 5.1).
SINGULAR_BRACKET_TOL = 1e-12

# J1, J2, J3 as (numerator quads, denominator quads) over tuple positions 0..5
_J_QUADS = (
    (((0, 1, 2, 3), (0, 1, 4, 5)), ((0, 1, 2, 4), (0, 1, 3, 5))),
    (((0, 2, 3, 4), (1, 2, 3, 5)), ((0, 2, 3, 5), (1, 2, 3, 4))),
    (((0, 1, 4, 5), (2, 3, 4, 5)), ((0, 2, 4, 5), (1, 3, 4, 5))),
)


def log_invariant_gradients(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int],
    normalize: bool = True,
) -> np.ndarray:
    """
    Exact gradients of log|J1|, log|J2|, log|J3| with respect to the 24
    coordinates of the six unit-normalized representatives (point-major,
    shape (3, 24)). Uses d log|det M| = tr(M^{-1} dM): the gradient with
    respect to column m of M is row m of M^{-1} (docs/MATHEMATICS.md, Lemma 3.1).
    The gradients are orthogonal to each point's radial direction because the
    invariants are scale-invariant per point. With normalize=False the
    gradient is taken at the given (non-unit) representatives.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert len(tuple_indices) == 6
    P = points_homogeneous[:, list(tuple_indices)]
    if normalize:
        P = normalize_to_unit_sphere(P)
    gradients = np.zeros((3, 24))
    for k, ((n1, n2), (d1, d2)) in enumerate(_J_QUADS):
        for quad, sign in ((n1, 1.0), (n2, 1.0), (d1, -1.0), (d2, -1.0)):
            M = P[:, list(quad)]
            if normalize and abs(np.linalg.det(M)) < SINGULAR_BRACKET_TOL:
                raise np.linalg.LinAlgError("numerically singular bracket")
            inverse = np.linalg.inv(M)
            for m, point in enumerate(quad):
                gradients[k, 4 * point:4 * point + 4] += sign * inverse[m]
    return gradients


def compute_invariant_noise_gain(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, int, int, int, int, int]
) -> float:
    """
    kappa_J(S) = max_k || grad log|J_k| || on unit representatives.

    For isotropic N(0, sigma^2) noise on the unit representatives,
    sigma * kappa_J is the first-order standard deviation of the worst
    log-invariant, and the median of max_k |Delta log J_k| lies between
    0.674 and 1.383 times sigma * kappa_J to first order
    (docs/MATHEMATICS.md, Proposition 5.1). It ranks tuples by their real
    noise amplification far better than L(S) (Spearman 0.995 vs 0.84,
    kappa_J conditioning experiment). kappa_J <= 8 / delta_J <= 2 sqrt(2 L(S)) (Proposition 5.2).
    Returns inf for a degenerate tuple.
    """
    try:
        gradients = log_invariant_gradients(points_homogeneous, tuple_indices)
    except np.linalg.LinAlgError:
        return float("inf")
    norms = np.linalg.norm(gradients, axis=1)
    return float(np.max(norms)) if np.all(np.isfinite(norms)) else float("inf")


def batch_invariant_noise_gain(points_homogeneous: np.ndarray, tuples: np.ndarray) -> np.ndarray:
    """
    Vectorized kappa_J for many tuples at once: (4, n) points, (T, 6) tuple
    indices -> (T,) gains. Same definition as compute_invariant_noise_gain
    (unit representatives, exact gradients via rows of M^{-1}). Uses the C++
    kernel (cpp_core/src/pencil.cpp, ~0.4 us per tuple) when built, otherwise a
    batched NumPy version (~25 us per tuple); the per-tuple loop takes ~170 us.
    Tuples with a bracket below SINGULAR_BRACKET_TOL get inf.
    """
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    tuples = np.asarray(tuples, dtype=np.int64).reshape(-1, 6)
    cpp = get_cpp_module()
    if cpp is not None and hasattr(cpp, "batch_invariant_noise_gain"):
        return np.asarray(cpp.batch_invariant_noise_gain(np.ascontiguousarray(points_homogeneous, dtype=float), tuples))
    return _batch_invariant_noise_gain_numpy(points_homogeneous, tuples)


def _batch_invariant_noise_gain_numpy(points_homogeneous: np.ndarray, tuples: np.ndarray) -> np.ndarray:
    """NumPy reference for batch_invariant_noise_gain (also used when the C++ module is absent)."""
    unit = normalize_to_unit_sphere(points_homogeneous)
    n_tuples = len(tuples)
    quads = sorted({q for (n1, n2), (d1, d2) in _J_QUADS for q in (n1, n2, d1, d2)})
    index = {q: k for k, q in enumerate(quads)}
    cols = tuples[:, np.array(quads)]                                   # (T, Q, 4) point indices
    M = np.transpose(unit[:, cols], (1, 2, 0, 3))                       # (T, Q, 4, 4)
    det = np.linalg.det(M)
    singular = np.abs(det) < SINGULAR_BRACKET_TOL
    M[singular] = np.eye(4)
    inv = np.linalg.inv(M)                                              # rows = column gradients
    gradients = np.zeros((n_tuples, 3, 6, 4))
    for k, ((n1, n2), (d1, d2)) in enumerate(_J_QUADS):
        for quad, sign in ((n1, 1.0), (n2, 1.0), (d1, -1.0), (d2, -1.0)):
            q = index[quad]
            for m, position in enumerate(quad):
                gradients[:, k, position, :] += sign * inv[:, q, m, :]
    gains = np.linalg.norm(gradients.reshape(n_tuples, 3, 24), axis=2).max(axis=1)
    gains[singular.any(axis=1)] = np.inf
    return gains
