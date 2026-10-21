"""
algorithms/degeneracy/coplanarity.py

Coplanarity (stratum-distance) score for a small set of two-view correspondences
(pre-registration docs/prereg/degeneracy_matched.md, Section 2; matched degeneracy experiment).

What is computed
----------------
Let u be the 4n pixel coordinates of n correspondences (n = 8 in matched degeneracy experiment) and X(u) a
projective reconstruction computed from them (unit columns). For a subset Q of
m >= 4 points, coplanarity of Q is the vanishing of every bracket of Q, i.e.
rank P_Q <= 3. With a base triple (a, b, c) of Q (the three points of Q with the
largest triangle in view 1), the m - 3 constraints are g_d = [a b c d],
d in Q minus {a, b, c}. The score is the standard Sampson (first-order)
approximation (Sampson 1982; Hartley-Zisserman 2nd ed. Sec. 4.2.6; cited) of the
squared pixel distance, in units of the noise, from u to the set of data whose
reconstruction by this estimator in these charts makes Q coplanar:

    T_Q = g^T (J J^T)^+ g / sigma_px^2,   J = dg/du (central differences).

T_Q is evaluated through the SVD of J with the rank and feasibility rules of
prereg Section 6.1 (numerical rank r = #{s_i > 1e-6 s_1}; if r < m - 3 the
linearised problem g + J du = 0 is feasible iff the part of g outside range(J) has
norm <= 1e-8 ||g||; feasible -> pseudo-inverse value, infeasible -> +inf). Nothing
here is claimed to be new: it is the Sampson approximation applied to bracket
constraints.

Two reconstructions are provided. (M8) The linear 8-point F in fixed image charts
T1, T2 (in matched degeneracy experiment the Hartley charts of the observed points, computed once and held
fixed while J is differenced), rank 2 by Frobenius projection, P1 = [I|0],
P2 = [[e2]x F | e2], DLT triangulation in the same charts, unit columns, signs
aligned to the unperturbed solution. (SC) Fixed cameras P1, P2 in fixed charts;
only the DLT triangulation depends on u.

Charts. capri_kappa and capri_sigma evaluated on the M8 reconstruction live in the
8-point Hartley reconstruction chart. Pixel noise is not isotropic in that chart,
so the noise reading of kappa_J (MATHEMATICS.md, Prop. 5.1) holds there only
qualitatively, and both scores are chart-dependent.

What is invariant and what depends on the chart (labels as in docs/MATHEMATICS.md)
------------------------------------------------------------------------------------
(1) Q coplanar (all brackets 0) is invariant under GL(4) x (R*)^n.  [proved]
    Proof: [A p_a l_a, ..., A p_d l_d] = det(A) l_a l_b l_c l_d [p_a ... p_d]
    (Eq. (1.1)); det(A) and the l_i are non-zero, so zero brackets stay zero.

(2) delta_m = min_{|Q|=m} sigma_4(unit columns of Q) is invariant under per-point
    scaling and orthogonal changes of frame; under A in GL(4) it changes by a factor
    in [1/kappa(A), kappa(A)], kappa = s_max(A)/s_min(A).  [proved]
    Proof: unit normalisation removes |l_i| and a sign flip of a column does not change
    singular values; orthogonal A keeps unit columns unit and singular values fixed.
    For general A, unit(A P) = A P D with D = diag(1/||A p_i||), entries in
    [1/s_max, 1/s_min]. By sigma_k(M N) <= sigma_1(M) sigma_k(N) and
    sigma_k(M N) >= sigma_min(M) sigma_k(N) for square invertible M (and the same with
    the diagonal factor on the right), s_min(A) sigma_4(P)/s_max <= sigma_4(A P D)
    <= s_max(A) sigma_4(P)/s_min. The bound holds for every Q, hence for the min.
    The magnitude of delta_m therefore depends on the reconstruction frame.

(3) delta_m is non-decreasing in m.  [proved]
    Proof: deleting a column cannot increase sigma_4 (interlacing for column
    deletion), and every (m+1)-subset contains an m-subset.

(4) Counting. 8 correspondences give 32 numbers. The two-view model has at most
    7 + 8 x 3 = 31 dof modulo PGL(4) (rank-2 F, then 3 dof per point given F), so its
    image in R^32 has dimension <= 31 and the reconstruction X(u) is a projection of
    the data onto the model, not a re-parametrisation; at least one residual dof is
    discarded (exactly one if the parametrisation is generically of full rank, which
    is the standard count of HZ Ch. 9-11; cited). The 8 linear epipolar equations plus
    det F = 0 over-determine F, which is why the linear F of 8 generic noisy points has
    det F != 0 (numerical check in tests/test_coplanarity.py).  [proved (counting)]

(5) T_Q is unchanged to first order when g -> D(u) g with D smooth and invertible:
    (i) a PGL(4) change applied to the estimator output, (ii) per-point rescaling of
    the representatives, (iii) another non-collinear base triple. Not the image chart.
    [proved (first order), for (i)-(iii) only]
    Proof: g' = D g gives J' = D J + (dD) g. Near a point u0 of the zero set where J has
    full rank m - 3, ||g|| = O(eps) for data at distance eps, so
    T' = g^T D^T (D J J^T D^T + O(eps))^{-1} D g = T (1 + O(eps)). (i), (ii): the
    brackets of unit(l_i A x_i) equal det(A) prod_i sign(l_i)/||A x_i|| times the old
    ones, a smooth non-vanishing diagonal D. (iii): for a non-collinear triple abc, Q is
    coplanar iff [abcd] = 0 for all d (abc spans a plane, d lies on it iff the bracket
    vanishes), so both constraint maps have the same zero set; d[abcd]/dd is the normal
    of plane abc, so each map is a submersion there, and by Hadamard's lemma in
    coordinates where one map is (y_1..y_r) the other equals D(u) y with D invertible
    on the stratum.
    Scope (post-hoc checker note, prereg Deviations): this is a first-order statement
    only, and it does not make the score representation-independent at finite noise.
    capri_coplanarity is therefore defined and claimed for unit columns and the
    largest-triangle base triple only. Checked qualitatively by
    test_affine_representatives_change_distance_beyond_first_order (affine
    representatives X/X_4 move T by more than 20% at 1e-3 px on a badly conditioned
    configuration). No effect size on the matched degeneracy experiment samples is claimed: the post-hoc
    numbers in the prereg come from scratch scripts and are not reproducible from
    repository code.  [numerical (in-project), qualitative only]

(6) T_Q depends on the image-normalisation chart (T1, T2) and on the estimator to
    first order. The estimator drops the residual dof of (4) in a chart-dependent way,
    so the set "data whose reconstruction makes Q coplanar" differs between charts off
    the 31-dim model; DLT with fixed cameras also drops one residual dof per point.
    Checked qualitatively by test_distance_changes_with_image_chart (Hartley chart vs an
    affine re-normalisation of it changes T by more than 1%). No chart-invariance is
    claimed; the score is defined in the stated charts.  [numerical (in-project),
    qualitative only]

(7) For coplanar truth, Gaussian pixel noise of s.d. sigma_px and a fixed chart,
    T_Q -> chi2_{m-3} as sigma -> 0.  [proved (first order), per fixed chart]
    Hypothesis: u -> g(X(u)) is smooth near the noise-free data u0 with J of full rank
    m - 3 there. This holds at k = 6 (the 8-point system has rank 8, item (8)) but
    fails at k >= 7, where noise-free data have no unique F and X(u) is undefined at
    the truth. Proof: noise-free data reconstruct exactly (item (8)), so g(u0) = 0 in
    every chart; g(u0 + sigma n) = sigma J n + O(sigma^2), hence
    T_Q = n^T J^T (J J^T)^{-1} J n + O(sigma), a rank-(m-3) orthogonal projection of a
    standard normal vector. A chart computed from the noisy data differs from the
    noise-free one by O(sigma), which changes J by O(sigma). Design exploration (scratch
    scripts, not reproducible from repository code; motivation, not evidence) showed a
    heavy tail at k = 6, where the 8-point system is close to rank 7 and higher order
    terms are large, and the generator's planar points lie within 8e-3 of the plane
    (make_scene), so chi2 thresholds are not trusted.

(8) Noise-free linear 8-point system: k <= 6 coplanar points (on the plane in general
    position, not on a conic) plus 8 - k generic points give rank 8; k = 7 gives rank
    <= 7, k = 8 rank <= 6, with null space inside {[e]x H}; every member of that null
    space reconstructs the k points as coplanar.  [proved, using the cited HZ Results
    13.1 and 13.3 on plane-induced homographies; numerical check]
    Proof: let H be the plane homography and G a null vector. On the 6 planar points
    x1^T H^T G x1 = 0, a conic through 6 points not on a conic, so the symmetric part of
    H^T G vanishes: H^T G = [a]x, G = H^{-T}[a]x = [H a]x H / det H. Each off-plane
    point gives a . (x1 x H^{-1} x2) = 0; the vectors x1 x H^{-1} x2 are epipolar lines
    through e1, independent for two generic points, so a is unique up to scale
    (rank 8). With 7 planar points one condition remains (2-dim null space), with 8
    none. A member F = [e]x H has det F = 0 (rank-2 projection is the identity), left
    epipole e, and H is compatible with F (H^T F is skew), so H = M - e v^T for the
    camera P2 = [M | e] (HZ 13.3) and an exact point with x2 ~ H x1 triangulates to a
    point of the plane (v^T, 1) (HZ 13.1). The generator's planar points are only near
    a plane, so its k >= 7 samples are near rank 7, not exactly rank 7.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.stats import chi2

from algorithms.utils import normalize_to_unit_sphere

RANK_RELATIVE_TOL = 1e-6        # s_i(J) > 1e-6 s_1(J): cond(J J^T) <= 1e12 (prereg 6.1)
FEASIBILITY_RELATIVE_TOL = 1e-8  # ||g outside range(J)|| <= 1e-8 ||g|| (prereg 6.1)
STATUS_FULL_RANK, STATUS_NONFINITE, STATUS_FEASIBLE_PINV, STATUS_INFEASIBLE = 0, 1, 2, 3


@lru_cache(maxsize=None)
def subsets_of_size(n_points: int, subset_size: int) -> np.ndarray:
    """All subsets of range(n_points) of the given size, (S, subset_size), lexicographic."""
    assert 1 <= subset_size <= n_points
    return np.array(list(combinations(range(n_points), subset_size)), dtype=np.int64)


def apply_image_chart(points_px: np.ndarray, chart: np.ndarray) -> np.ndarray:
    """(..., n, 2) pixels -> (..., n, 3) homogeneous chart coordinates chart @ (x, y, 1)."""
    assert points_px.shape[-1] == 2 and chart.shape == (3, 3)
    homogeneous = np.concatenate([points_px, np.ones(points_px.shape[:-1] + (1,))], axis=-1)
    return homogeneous @ chart.T


def eight_point_design_matrix(chart_1_points: np.ndarray, chart_2_points: np.ndarray) -> np.ndarray:
    """Rows x2_i x1_j of the linear epipolar system x2^T F x1 = 0: (B, n, 3) x2 -> (B, n, 9)."""
    assert chart_1_points.shape == chart_2_points.shape and chart_1_points.shape[-1] == 3
    outer = np.einsum("...ni,...nj->...nij", chart_2_points, chart_1_points)
    return outer.reshape(outer.shape[:-2] + (9,))


def linear_fundamental_matrix(design: np.ndarray) -> np.ndarray:
    """Right singular vector of the smallest singular value, (B, n, 9) -> (B, 3, 3); no rank-2 step."""
    assert design.ndim == 3 and design.shape[2] == 9 and design.shape[1] >= 8
    return np.linalg.svd(design)[2][:, -1].reshape(-1, 3, 3)


def project_to_rank_two(fundamental: np.ndarray) -> np.ndarray:
    """Nearest rank-2 matrix in the Frobenius norm (smallest singular value set to 0), batched."""
    assert fundamental.shape[-2:] == (3, 3)
    left, singular, right = np.linalg.svd(fundamental)
    singular[..., 2] = 0.0
    return left @ (singular[..., :, None] * right)


def align_signs(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip each vector (last axis) so that its dot product with the reference is >= 0."""
    assert vectors.shape[-reference.ndim:] == reference.shape
    dots = np.sum(vectors * reference, axis=-1, keepdims=True)
    return vectors * np.where(dots < 0, -1.0, 1.0)


def canonical_camera_pair(fundamental: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P1 = [I | 0], P2 = [[e2]x F | e2] with F^T e2 = 0 (HZ Result 9.14), e2 unit, batched (B, 3, 3)."""
    assert fundamental.ndim == 3 and fundamental.shape[1:] == (3, 3)
    epipole = np.linalg.svd(fundamental)[0][:, :, 2]
    epipole = align_signs(epipole, epipole[0])
    skew = np.zeros_like(fundamental)
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -epipole[:, 2], epipole[:, 1], -epipole[:, 0]
    skew[:, 1, 0], skew[:, 2, 0], skew[:, 2, 1] = epipole[:, 2], -epipole[:, 1], epipole[:, 0]
    camera_2 = np.concatenate([skew @ fundamental, epipole[:, :, None]], axis=2)
    camera_1 = np.broadcast_to(np.hstack([np.eye(3), np.zeros((3, 1))]), camera_2.shape)
    return camera_1, camera_2


def batch_dlt_triangulation(chart_1_points: np.ndarray, chart_2_points: np.ndarray,
                            camera_1: np.ndarray, camera_2: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation as in triangulate_point_dlt, batched: (B, n, 3) x2 and
    (B, 3, 4) or (3, 4) cameras -> (B, 4, n) unit columns, signs aligned to batch item 0."""
    assert chart_1_points.shape == chart_2_points.shape and chart_1_points.shape[-1] == 3
    camera_1 = np.broadcast_to(camera_1, (chart_1_points.shape[0], 3, 4))
    camera_2 = np.broadcast_to(camera_2, (chart_1_points.shape[0], 3, 4))

    def dlt_rows(camera: np.ndarray, points: np.ndarray) -> np.ndarray:
        third = camera[:, None, 2, :]
        return np.stack([points[..., 0:1] * third - points[..., 2:3] * camera[:, None, 0, :],
                         points[..., 1:2] * third - points[..., 2:3] * camera[:, None, 1, :]], axis=2)

    system = np.concatenate([dlt_rows(camera_1, chart_1_points), dlt_rows(camera_2, chart_2_points)], axis=2)
    points_3d = np.linalg.svd(system)[2][..., -1, :]
    points_3d /= np.linalg.norm(points_3d, axis=2, keepdims=True)
    return np.transpose(align_signs(points_3d, points_3d[0]), (0, 2, 1))


def batch_eight_point_reconstruction(view_1_px: np.ndarray, view_2_px: np.ndarray, chart_1: np.ndarray,
                                     chart_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M8 reconstruction of B perturbed copies of n >= 8 correspondences (B, n, 2) in fixed charts:
    linear F, Frobenius rank 2, canonical cameras, DLT. Returns (unit points (B, 4, n), F (B, 3, 3))."""
    assert view_1_px.shape == view_2_px.shape and view_1_px.ndim == 3 and view_1_px.shape[1] >= 8
    points_1, points_2 = apply_image_chart(view_1_px, chart_1), apply_image_chart(view_2_px, chart_2)
    fundamental = project_to_rank_two(linear_fundamental_matrix(eight_point_design_matrix(points_1, points_2)))
    fundamental = align_signs(fundamental.reshape(-1, 9), fundamental[0].ravel()).reshape(-1, 3, 3)
    camera_1, camera_2 = canonical_camera_pair(fundamental)
    return batch_dlt_triangulation(points_1, points_2, camera_1, camera_2), fundamental


def batch_fixed_camera_triangulation(view_1_px: np.ndarray, view_2_px: np.ndarray, chart_1: np.ndarray,
                                     chart_2: np.ndarray, camera_1: np.ndarray, camera_2: np.ndarray) -> np.ndarray:
    """SC reconstruction: fixed cameras (3, 4) in fixed charts, DLT of (B, n, 2) -> (B, 4, n) unit."""
    assert camera_1.shape == (3, 4) and camera_2.shape == (3, 4)
    return batch_dlt_triangulation(apply_image_chart(view_1_px, chart_1), apply_image_chart(view_2_px, chart_2),
                                   camera_1, camera_2)


def perturbed_correspondences(view_1_px: np.ndarray, view_2_px: np.ndarray,
                              step_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Copies of the data for central differences: item 0 unperturbed, items 1..4n add +step to
    coordinate i of u = (view_1.ravel(), view_2.ravel()), items 4n+1..8n add -step."""
    assert view_1_px.shape == view_2_px.shape and view_1_px.shape[1] == 2 and step_px > 0
    n_coordinates = 4 * len(view_1_px)
    data = np.concatenate([view_1_px.ravel(), view_2_px.ravel()])
    offsets = np.concatenate([np.zeros((1, n_coordinates)), step_px * np.eye(n_coordinates),
                              -step_px * np.eye(n_coordinates)])
    copies = data[None] + offsets
    half = n_coordinates // 2
    return copies[:, :half].reshape(-1, len(view_1_px), 2), copies[:, half:].reshape(-1, len(view_1_px), 2)


def largest_triangle_base_triples(view_1_px: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """For each subset (S, m), the three of its points with the largest triangle in view 1, (S, 3);
    ties go to the first triple in lexicographic order."""
    assert view_1_px.ndim == 2 and view_1_px.shape[1] == 2 and subsets.shape[1] >= 3
    triples = subsets[:, subsets_of_size(subsets.shape[1], 3)]                     # (S, C, 3)
    corners = np.concatenate([view_1_px[triples], np.ones(triples.shape + (1,))], axis=3)
    areas = np.abs(np.linalg.det(corners))
    return triples[np.arange(len(subsets)), np.argmax(areas, axis=1)]


def bracket_constraints(points_unit: np.ndarray, subsets: np.ndarray, base_triples: np.ndarray) -> np.ndarray:
    """g_d = [a b c d] for d in Q minus the base triple, in subset order: (B, 4, n) -> (B, S, m - 3)."""
    assert points_unit.ndim == 3 and points_unit.shape[1] == 4 and base_triples.shape == (len(subsets), 3)
    remaining = np.array([[d for d in subset if d not in triple] for subset, triple in zip(subsets, base_triples)])
    quads = np.concatenate([np.repeat(base_triples[:, None, :], remaining.shape[1], axis=1),
                            remaining[:, :, None]], axis=2)                          # (S, m - 3, 4)
    matrices = np.moveaxis(points_unit[:, :, quads], 1, -2)                          # (B, S, m - 3, 4, 4)
    return np.linalg.det(matrices)


def sampson_distances(constraints: np.ndarray, step_px: float, sigma_px: float) -> tuple[np.ndarray, np.ndarray]:
    """T_Q = g^T (J J^T)^+ g / sigma^2 per subset from constraints on the perturbed copies
    (B = 1 + 2N, S, r), with J by central differences and the prereg 6.1 rules. Returns
    (T (S,), status (S,)) with status in STATUS_*; never raises on singular or non-finite input."""
    assert constraints.ndim == 3 and constraints.shape[0] % 2 == 1 and step_px > 0 and sigma_px > 0
    n_coordinates = (constraints.shape[0] - 1) // 2
    value = constraints[0]                                                           # (S, r)
    jacobian = np.moveaxis((constraints[1:1 + n_coordinates] - constraints[1 + n_coordinates:]) / (2 * step_px),
                           0, -1)                                                    # (S, r, N)
    finite = np.all(np.isfinite(value), axis=1) & np.all(np.isfinite(jacobian), axis=(1, 2))
    distances = np.full(len(value), np.nan)
    status = np.full(len(value), STATUS_NONFINITE, dtype=np.int64)
    if not np.any(finite):
        return distances, status
    left, singular, _ = np.linalg.svd(jacobian[finite], full_matrices=False)
    coefficients = np.einsum("sji,sj->si", left, value[finite])                     # U^T g
    in_range = singular > RANK_RELATIVE_TOL * singular[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        quadratic = np.sum(np.where(in_range, coefficients ** 2 / np.where(in_range, singular, 1.0) ** 2, 0.0), axis=1)
    full_rank = np.all(in_range, axis=1)
    outside = np.linalg.norm(np.where(in_range, 0.0, coefficients), axis=1)
    feasible = outside <= FEASIBILITY_RELATIVE_TOL * np.linalg.norm(value[finite], axis=1)
    distances[finite] = np.where(full_rank | feasible, quadratic, np.inf) / sigma_px ** 2
    status[finite] = np.where(full_rank, STATUS_FULL_RANK, np.where(feasible, STATUS_FEASIBLE_PINV, STATUS_INFEASIBLE))
    return distances, status


def stratum_distance_profile(perturbed_points_unit: np.ndarray, view_1_px: np.ndarray, subset_sizes: tuple[int, ...],
                             step_px: float, sigma_px: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{m: (T_Q, status) over all m-subsets in lexicographic order} for the reconstructions of the
    perturbed copies (B, 4, n); base triples from the observed view-1 pixels."""
    n_points = perturbed_points_unit.shape[2]
    assert view_1_px.shape == (n_points, 2) and all(4 <= m <= n_points for m in subset_sizes)
    profile = {}
    for subset_size in subset_sizes:
        subsets = subsets_of_size(n_points, subset_size)
        constraints = bracket_constraints(perturbed_points_unit, subsets, largest_triangle_base_triples(view_1_px, subsets))
        profile[subset_size] = sampson_distances(constraints, step_px, sigma_px)
    return profile


def coplanarity_score(distances: np.ndarray, status: np.ndarray, infeasible_distance: float = np.inf) -> float:
    """-log10(min_Q T_Q), higher = more degenerate. NaN subsets are left out (nanmin); infeasible
    subsets enter with `infeasible_distance` (+inf primary, 0 in the opposite rule). Returns +inf when
    the min is 0, -inf when it is +inf, NaN when every subset is NaN (mapped by the caller)."""
    assert distances.shape == status.shape
    values = np.where(status == STATUS_INFEASIBLE, infeasible_distance, distances)
    if np.all(np.isnan(values)):
        return float("nan")
    with np.errstate(divide="ignore"):
        return float(-np.log10(np.nanmin(values)))


def estimate_stratum_size(min_distance_by_size: dict[int, float], level: float = 0.99) -> int:
    """m_hat = max{m in 5..n : min_{|Q|=m} T_Q <= chi2_{m-3}(level)}, or 4 meaning '<= 4'.
    The chi2 law holds only to first order and only where its hypothesis holds (item (7))."""
    assert 0.0 < level < 1.0 and all(m >= 5 for m in min_distance_by_size)
    accepted = [m for m, value in min_distance_by_size.items() if value <= chi2.ppf(level, m - 3)]
    return max(accepted) if accepted else 4


def coplanarity_gap(points_homogeneous: np.ndarray, subset_size: int) -> float:
    """delta_m = min over m-subsets of sigma_4 of the unit columns (chart-dependent, item (2)).
    Generalises largest_coplanar_subset_size (stratification.py) from 6 to n points."""
    assert points_homogeneous.ndim == 2 and points_homogeneous.shape[0] == 4
    assert 4 <= subset_size <= points_homogeneous.shape[1]
    unit = normalize_to_unit_sphere(points_homogeneous)
    matrices = np.moveaxis(unit[:, subsets_of_size(unit.shape[1], subset_size)], 0, 1)   # (S, 4, m)
    return float(np.min(np.linalg.svd(matrices, compute_uv=False)[:, 3]))
