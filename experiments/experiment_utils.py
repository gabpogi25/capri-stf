"""
experiments/experiment_utils.py

Shared infrastructure for all CAPRI experiments.
Implements the global setup requirements from the Master Experiment Specification:
  §0.1 Hardware/software logging
  §0.2 Master seeding policy
  §0.3 Canonical regime generation (delegates to synthetic_data.generator)
  §0.4 Exact degeneracy strata generation
  §0.6 Standard noise grid
  §0.7 Bootstrap confidence intervals
  §0.8 Anti-circularity variance check
  §0.9 Statistical acceptance thresholds
"""

from __future__ import annotations
import os
import sys
import platform
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import Callable

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.invariants.brackets import compute_all_brackets_for_tuple, compute_bracket
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.syzygy.residual import compute_syzygy_residual, compute_drift_threshold, is_tuple_drift_admissible
from algorithms.conditioning.scores import compute_conditioning_score, compute_delta_min
from algorithms.degeneracy.stratification import classify_degeneracy_stratum, DegeneracyStratum
from algorithms.selection.greedy import select_tuples_greedy
from algorithms.utils import normalize_to_unit_sphere, embed_affine_to_homogeneous
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
    generate_mismatched_configuration,
    generate_non_rigid_configuration,
    apply_random_projective_transform,
)


# ============================================================
# §0.1 Hardware / Software Logging
# ============================================================

def log_hardware_info() -> dict:
    """Capture exact CPU/GPU/library versions per §0.1."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except ImportError:
        info["scipy_version"] = "NOT INSTALLED"
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except (ImportError, OSError):
        info["torch_version"] = "NOT INSTALLED OR BLOCKED"
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
    except ImportError:
        info["opencv_version"] = "NOT INSTALLED"
    return info


# ============================================================
# §0.2 Master Seeding Policy
# ============================================================

BASE_SEED = 42

def get_experiment_seed(experiment_id: int) -> int:
    """Per-experiment seed: seed_i = BASE_SEED * 1000 + i."""
    return BASE_SEED * 1000 + experiment_id

def get_finding_seed(finding_id: int) -> int:
    """Per-finding seed: BASE_SEED * 1000 + 100 + finding_id."""
    return BASE_SEED * 1000 + 100 + finding_id

def make_rng(seed: int) -> np.random.Generator:
    """Create an explicit RNG instance (never use global np.random.seed)."""
    return np.random.default_rng(seed)


# ============================================================
# §0.4 Exact Degeneracy Strata Generation
# ============================================================

def generate_generic_stratum(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n_points in fully generic position. Returns (4, n_points)."""
    return generate_generic_configuration(n_points, scale=1.0, rng=rng)


def generate_boundary_stratum(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Force exactly one 4-subset to be exactly coplanar (bracket = 0 to machine precision).
    Remaining points are generic. codim = 1.
    Returns (4, n_points).
    """
    assert n_points >= 6
    # Start with generic configuration
    pts_3d = rng.uniform(-1.0, 1.0, size=(n_points, 3))

    # Pick a random 4-subset to make coplanar
    coplanar_indices = list(rng.choice(n_points, size=4, replace=False))

    # Define a random plane through the first point
    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    anchor = pts_3d[coplanar_indices[0]]
    plane_d = np.dot(anchor, normal)

    # Project the 4 chosen points exactly onto this plane
    for idx in coplanar_indices:
        dist = np.dot(pts_3d[idx], normal) - plane_d
        pts_3d[idx] -= dist * normal

    return embed_affine_to_homogeneous(pts_3d)


def generate_compound_stratum(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Force one 5-subset to be exactly coplanar. codim = 2.
    Returns (4, n_points).
    """
    assert n_points >= 6
    pts_3d = rng.uniform(-1.0, 1.0, size=(n_points, 3))

    coplanar_indices = list(rng.choice(n_points, size=5, replace=False))

    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    anchor = pts_3d[coplanar_indices[0]]
    plane_d = np.dot(anchor, normal)

    for idx in coplanar_indices:
        dist = np.dot(pts_3d[idx], normal) - plane_d
        pts_3d[idx] -= dist * normal

    return embed_affine_to_homogeneous(pts_3d)


def generate_critical_stratum(n_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Force all 6 points (first 6) to be exactly coplanar. codim = 3.
    Returns (4, n_points).
    """
    assert n_points >= 6
    pts_3d = rng.uniform(-1.0, 1.0, size=(n_points, 3))

    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    anchor = pts_3d[0]
    plane_d = np.dot(anchor, normal)

    # Force first 6 points exactly onto the plane
    for idx in range(min(6, n_points)):
        dist = np.dot(pts_3d[idx], normal) - plane_d
        pts_3d[idx] -= dist * normal

    return embed_affine_to_homogeneous(pts_3d)


def generate_stratum_config(stratum: str, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Dispatch to the correct stratum generator.
    stratum ∈ {"generic", "boundary", "compound", "critical"}
    """
    generators = {
        "generic": generate_generic_stratum,
        "boundary": generate_boundary_stratum,
        "compound": generate_compound_stratum,
        "critical": generate_critical_stratum,
    }
    if stratum not in generators:
        raise ValueError(f"Unknown stratum: {stratum}. Must be one of {list(generators.keys())}")
    return generators[stratum](n_points, rng)


def generate_near_degenerate_config(
    n_points: int, kappa: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Generate a config where a 4-subset is pulled toward (but not onto) a plane.
    kappa ∈ [0, 1]: 0 = fully generic, 1 = exactly coplanar.
    """
    pts_3d = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    coplanar_indices = list(rng.choice(n_points, size=4, replace=False))

    normal = rng.standard_normal(3)
    normal /= np.linalg.norm(normal)
    anchor = pts_3d[coplanar_indices[0]]
    plane_d = np.dot(anchor, normal)

    for idx in coplanar_indices:
        dist = np.dot(pts_3d[idx], normal) - plane_d
        pts_3d[idx] -= kappa * dist * normal  # Pull toward plane by fraction kappa

    return embed_affine_to_homogeneous(pts_3d)


# ============================================================
# §0.4b Measurement model for the consistency residual
# ============================================================
#
# Brackets recomputed from one coherent (for example mismatched)
# configuration give rho = 0 by Corollary 2.6 (docs/MATHEMATICS.md), and
# averaging the brackets of two configurations has no physical meaning.
#
# Every regime therefore yields two independent measurements of the same six
# points (see synthetic_data.generator.estimate_brackets_independently).
# All regimes carry the same Gaussian measurement noise, so Generic is a fair
# null for Mismatched and Non-Rigid.

MEASUREMENT_NOISE_SIGMA = 1e-3


def regime_measurements(
    regime: str,
    n_points: int,
    rng: np.random.Generator,
    noise_sigma: float = MEASUREMENT_NOISE_SIGMA,
    mismatch_fraction: float = 0.15,
    deformation_bound: float = 0.05,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """
    Two measurements (A, B) of one synthetic scene for the given regime.

    Returns:
        (observed points used for L(S) and strata = measurement A's scene,
         [measurement_A, measurement_B] in the unit chart,
         indices of corrupted points (mismatched regime only, else []))
    """
    from synthetic_data.generator import (
        generate_sparse_view_configuration,
        measure_with_gaussian_noise,
    )
    corrupted: list[int] = []
    if regime == "Generic":
        scene_a = scene_b = generate_generic_configuration(n_points, scale=1.0, rng=rng)
    elif regime == "Near-Degenerate":
        scene_a = scene_b = generate_near_degenerate_configuration(
            n_points, coplanar_fraction=0.5, plane_offset=1e-3, scale=1.0, rng=rng)
    elif regime == "Mismatched":
        scene_a, scene_b, corrupted = generate_mismatched_configuration(
            n_points, mismatch_fraction=mismatch_fraction, scale=1.0, rng=rng)
        corrupted = [int(i) for i in corrupted]
    elif regime == "Non-Rigid":
        scene_a, scene_b, _ = generate_non_rigid_configuration(
            n_points, deformation_bound=deformation_bound, scale=1.0, rng=rng)
    elif regime == "Sparse-View":
        scene_a, _ = generate_sparse_view_configuration(
            n_points, n_views=2, points_per_view=6, scale=1.0, rng=rng)
        scene_b = scene_a
    else:
        raise ValueError(f"Unknown regime: {regime}")
    measurement_a = measure_with_gaussian_noise(scene_a, noise_sigma, rng)
    measurement_b = measure_with_gaussian_noise(scene_b, noise_sigma, rng)
    return scene_a, [measurement_a, measurement_b], corrupted


# ============================================================
# §0.5 Standard point-count grid
# ============================================================

STANDARD_N_GRID = list(range(10, 101, 5))  # 19 values: 10, 15, ..., 100


# ============================================================
# §0.6 Standard noise/perturbation grid
# ============================================================

STANDARD_NOISE_GRID = np.logspace(-4, -1, 20)  # 20 log-spaced levels from 1e-4 to 1e-1


# ============================================================
# §0.7 Bootstrap Confidence Intervals
# ============================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable = np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    rng: np.random.Generator = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    data = np.asarray(data)
    point_estimate = float(statistic(data))

    bootstrap_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        resample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats[i] = statistic(resample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


# ============================================================
# §0.8 Anti-Circularity Check
# ============================================================

def anti_circularity_check(
    data_per_trial: np.ndarray,
    label: str = "",
    min_variance: float = 1e-20,
) -> bool:
    """
    Verify that repeated trials at the same nominal parameter produce nonzero variance.
    If variance is below min_variance, the data may be formula-generated, not simulated.
    Returns True if the check passes (real stochastic variance detected).
    """
    variance = float(np.var(data_per_trial))
    passed = variance > min_variance
    if not passed:
        print(f"  [ANTI-CIRCULARITY FAIL] {label}: variance = {variance:.2e} < {min_variance:.2e}")
        print(f"    This suggests the data may be generated from a deterministic formula.")
    return passed


# ============================================================
# §0.9 Statistical Acceptance Thresholds
# ============================================================

def fit_power_law_loglog(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit y = C * x^k on log-log scale. Returns exponent k, intercept log(C),
    R², and the fit residuals.
    """
    # Filter out zero/negative values before log to avoid -inf outliers
    positive_mask = (x > 0) & (y > 0)
    x = x[positive_mask]
    y = y[positive_mask]

    if len(x) < 3:
        return {"exponent": np.nan, "intercept": np.nan, "r_squared": 0.0, "residuals": np.array([])}

    log_x = np.log10(x)
    log_y = np.log10(y)

    # Remove any inf/nan
    valid = np.isfinite(log_x) & np.isfinite(log_y)
    log_x = log_x[valid]
    log_y = log_y[valid]

    if len(log_x) < 3:
        return {"exponent": np.nan, "intercept": np.nan, "r_squared": 0.0, "residuals": np.array([])}

    coeffs = np.polyfit(log_x, log_y, 1)
    k = coeffs[0]
    intercept = coeffs[1]

    predicted = k * log_x + intercept
    ss_res = np.sum((log_y - predicted) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-30)

    residuals = log_y - predicted

    return {
        "exponent": float(k),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "residuals": residuals,
    }


def bootstrap_exponent_ci(
    x: np.ndarray, y: np.ndarray,
    n_resamples: int = 2000,
    rng: np.random.Generator = None,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI on the power-law exponent from log-log fit.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    exponents = []
    n = len(x)
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        result = fit_power_law_loglog(x[idx], y[idx])
        if np.isfinite(result["exponent"]):
            exponents.append(result["exponent"])

    if len(exponents) < 100:
        return (np.nan, np.nan, np.nan)

    exponents = np.array(exponents)
    point = float(np.mean(exponents))
    ci_lo = float(np.percentile(exponents, 2.5))
    ci_hi = float(np.percentile(exponents, 97.5))
    return point, ci_lo, ci_hi


# ============================================================
# Helpers
# ============================================================

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def save_results(df: pd.DataFrame, filename: str, seed: int, experiment_name: str) -> Path:
    """
    Save a DataFrame to CSV with a metadata header.

    The header records the seed, the numerical backend (C++ or Python, see
    algorithms/backend.py), the git commit and library versions, so every
    number can be traced to the code and implementation that produced it.
    Timestamps are deliberately omitted so that re-running an experiment
    with the same code reproduces the file byte for byte.
    """
    from algorithms.backend import backend_name
    filepath = RESULTS_DIR / filename
    from forensics.io.provenance import git_commit
    commit = git_commit()
    with open(filepath, 'w', newline="") as f:
        f.write(f"# Experiment: {experiment_name}\n")
        f.write(f"# Seed: {seed}\n")
        f.write(f"# Backend: {backend_name()}\n")
        f.write(f"# Code commit: {commit}\n")
        f.write(f"# Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}\n")
        df.to_csv(f, index=False)
    print(f"  Saved: {filepath}")
    return filepath


def pick_first_six_tuple(n_points: int) -> tuple:
    """Return the canonical first six-point tuple (0,1,2,3,4,5)."""
    return tuple(range(min(6, n_points)))


def random_six_tuple(n_points: int, rng: np.random.Generator) -> tuple:
    """Return a random six-point tuple."""
    return tuple(sorted(rng.choice(n_points, size=6, replace=False)))
