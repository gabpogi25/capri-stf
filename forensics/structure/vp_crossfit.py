"""
forensics/structure/vp_crossfit.py

Cross-fitted vanishing-point test (constraint: images of 3D-parallel lines
meet at one vanishing point).

Selection bias: if lines are grouped into a family BECAUSE they nearly meet
at one point and the same lines are then tested for meeting at one point, the
test is biased toward "consistent". Cross-fitting breaks the circle: the
vanishing points are estimated on fold A and every fold-B segment is tested
against them; then the folds swap.

Per test segment j assigned to family f the residual is
    e_j = wrap(theta_j - phi_f(m_j)),  phi_f(m) = direction from m to v_f,
and z_j = e_j / sqrt(sigma_theta_j^2 + Var(phi_f(m_j)) + (sigma_perp_j / |v_f - m_j|)^2).

The vanishing point is refined by maximum likelihood on the angular
residuals and Var(phi_f(m_j)) comes from its Gauss-Newton covariance by the
delta method. (A bootstrap of an algebraic fit is not used: the algebraic
fit is biased under this noise model and biases the z-scores, see
tests/test_forensics/test_properties.py::test_t08.)
Under the authentic-camera model z_j is approximately N(0, 1); truncation at
assign_angle_deg makes the test conservative. This is a hypothesis to be
checked on real authentic images (the real-media null gate), not a guarantee.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from forensics.structure.lines import Segments


@dataclass
class VanishingFamily:
    vp: np.ndarray                  # homogeneous (3,), unit norm
    fit_index: np.ndarray           # indices (within the fitting fold) of inlier segments
    covariance: np.ndarray          # (3, 3) covariance of vp (rank 2, tangent to the unit sphere)


TRIM_Z = 4.0    # standardized residual above which a fitting member is treated as a stray
RANSAC_Z = 5.0  # RANSAC consensus band in units of sigma_theta (capped at inlier_angle_deg)


def _wrap_half_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap to (-pi/2, pi/2] (line orientations are defined modulo pi)."""
    return (np.asarray(angle) + np.pi / 2) % np.pi - np.pi / 2


def direction_to_vp(vp: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Orientation (mod pi) of the line from each point to the homogeneous vanishing point."""
    d = vp[None, :2] - vp[2] * points
    return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)


def angular_residual(segments: Segments, vp: np.ndarray) -> np.ndarray:
    return _wrap_half_pi(segments.theta - direction_to_vp(vp, segments.midpoints))


def refine_vp(lines: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Algebraic least-squares VP (initial value): smallest right singular vector of the weighted lines."""
    _, _, vt = np.linalg.svd(lines * weights[:, None])
    vp = vt[-1]
    return vp / np.linalg.norm(vp)


def _tangent_basis(vp: np.ndarray) -> np.ndarray:
    """Two orthonormal vectors spanning the tangent plane of the unit sphere at vp (3 x 2)."""
    _, _, vt = np.linalg.svd(vp[None, :])
    return vt[1:].T


def ml_refine_vp(segments: Segments, initial_vp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Maximum-likelihood VP under the segment noise model: minimize
    sum_j (e_j(v) / s_j)^2 with e_j the angular residual and
    s_j^2 = sigma_theta_j^2 + (sigma_perp_j / |v - m_j|)^2 (held at the initial v).
    The algebraic fit is biased (errors-in-variables), which biased the z-scores.
    v is parameterized by two tangent coordinates at the initial estimate.
    Returns (vp, covariance of vp) with the Gauss-Newton covariance (J^T J)^-1
    mapped back to R^3.
    """
    basis = _tangent_basis(initial_vp)

    def vp_from(params: np.ndarray) -> np.ndarray:
        v = initial_vp + basis @ params
        return v / np.linalg.norm(v)

    vp0_point = initial_vp[:2] / initial_vp[2] if abs(initial_vp[2]) > 1e-12 else None
    if vp0_point is None:
        perp = np.zeros(len(segments))
    else:
        perp = segments.sigma_perp / np.maximum(np.linalg.norm(segments.midpoints - vp0_point, axis=1), 1e-12)
    scale = np.sqrt(segments.sigma_theta ** 2 + perp ** 2)

    def residuals(params: np.ndarray) -> np.ndarray:
        return angular_residual(segments, vp_from(params)) / scale

    solution = least_squares(residuals, np.zeros(2), method="lm")
    vp = vp_from(solution.x)
    jacobian = solution.jac
    try:
        cov_params = np.linalg.inv(jacobian.T @ jacobian)
    except np.linalg.LinAlgError:
        cov_params = np.full((2, 2), np.inf)
    # d vp / d params at the solution (normalization included), by finite differences.
    h = 1e-7
    d_vp = np.column_stack([(vp_from(solution.x + h * np.eye(2)[k]) - vp_from(solution.x - h * np.eye(2)[k])) / (2 * h)
                            for k in range(2)])
    return vp, d_vp @ cov_params @ d_vp.T


def ransac_families(segments: Segments, config: dict, rng: np.random.Generator) -> list[VanishingFamily]:
    """Sequential RANSAC for up to k_max vanishing points on one fold."""
    cfg = config["vanishing_points"]
    lines = segments.homogeneous_lines
    remaining = np.arange(len(segments))
    families: list[VanishingFamily] = []
    inlier_angle = np.deg2rad(cfg["inlier_angle_deg"])
    for _ in range(cfg["k_max_families"]):
        if len(remaining) < max(2, cfg["min_fit_lines"]):
            break
        best_inliers = np.array([], dtype=int)
        for _ in range(cfg["ransac_iters"]):
            i, j = rng.choice(remaining, size=2, replace=False)
            vp = np.cross(lines[i], lines[j])
            norm = np.linalg.norm(vp)
            if norm < 1e-12:
                continue
            residual = np.abs(angular_residual(segments.subset(remaining), vp / norm))
            band = np.minimum(inlier_angle, RANSAC_Z * segments.sigma_theta[remaining])
            inliers = remaining[residual <= band]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
        if len(best_inliers) < cfg["min_fit_lines"]:
            break
        vp = refine_vp(lines[best_inliers], segments.length_px[best_inliers])
        members = best_inliers
        # RANSAC's angular inlier band (degrees) is wide compared with sigma_theta
        # of long segments (~0.1 deg), so it can admit segments of another
        # family. Refit by ML and drop members whose standardized residual
        # exceeds TRIM_Z, until the member set is stable.
        for _ in range(5):
            vp, covariance = ml_refine_vp(segments.subset(members), vp)
            standardized = np.abs(angular_residual(segments.subset(members), vp)) / segments.sigma_theta[members]
            keep = standardized <= TRIM_Z
            if keep.all() or keep.sum() < cfg["min_fit_lines"]:
                break
            members = members[keep]
        families.append(VanishingFamily(vp=vp, fit_index=members, covariance=covariance))
        remaining = np.setdiff1d(remaining, best_inliers)
    return families


def test_fold(test: Segments, families: list[VanishingFamily], config: dict) -> list[dict]:
    """z-scores of test segments assigned (angle <= assign_angle_deg) to the nearest family."""
    if not families:
        return []
    assign_angle = np.deg2rad(config["vanishing_points"]["assign_angle_deg"])
    residuals = np.stack([angular_residual(test, f.vp) for f in families])     # (F, n)
    nearest = np.argmin(np.abs(residuals), axis=0)
    candidates = np.sum(np.abs(residuals) <= assign_angle, axis=0)
    out = []
    for j in range(len(test)):
        f = nearest[j]
        e = residuals[f, j]
        # Gate: no family within the band, or more than one (ambiguous
        # assignment; a wrong family would give a spuriously huge z).
        if abs(e) > assign_angle or candidates[j] > 1:
            continue
        family = families[f]
        midpoint = test.midpoints[j:j + 1]
        reference = direction_to_vp(family.vp, midpoint)[0]
        h = 1e-7
        gradient = np.array([_wrap_half_pi(direction_to_vp(family.vp + h * np.eye(3)[k], midpoint)[0]
                                           - direction_to_vp(family.vp - h * np.eye(3)[k], midpoint)[0]) / (2 * h)
                             for k in range(3)])
        var_boot = float(gradient @ family.covariance @ gradient)   # delta-method variance of phi(m_j)
        vp_point = family.vp[:2] / family.vp[2] if abs(family.vp[2]) > 1e-12 else None
        distance = np.inf if vp_point is None else float(np.linalg.norm(vp_point - midpoint[0]))
        var_perp = (test.sigma_perp[j] / distance) ** 2 if np.isfinite(distance) and distance > 0 else 0.0
        variance = test.sigma_theta[j] ** 2 + var_boot + var_perp
        out.append({"family": int(f), "residual": float(e), "z": float(e / np.sqrt(variance)),
                    "variance": float(variance), "length_px": float(test.length_px[j])})
    return out


def split_folds(segments: Segments, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random halves, stratified by length tercile."""
    order = np.argsort(segments.length_px)
    fold_a, fold_b = [], []
    for tercile in np.array_split(order, 3):
        shuffled = rng.permutation(tercile)
        half = len(shuffled) // 2
        fold_a.extend(shuffled[:half])
        fold_b.extend(shuffled[half:])
    return np.asarray(fold_a, dtype=int), np.asarray(fold_b, dtype=int)


def cross_fitted_vp_test(segments: Segments, config: dict, rng: np.random.Generator) -> list[dict]:
    """Fit on A, test on B, swap; pooled per-segment results (each segment tested once)."""
    fold_a, fold_b = split_folds(segments, rng)
    results = []
    for fit_idx, test_idx in ((fold_a, fold_b), (fold_b, fold_a)):
        families = ransac_families(segments.subset(fit_idx), config, rng)
        for record in test_fold(segments.subset(test_idx), families, config):
            results.append({**record, "kind": "vp_angle"})
    return results
