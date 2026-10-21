"""
forensics/pipeline_image.py

Single-image structural branch.

    decode at native resolution -> LSD segments with noise model
    -> cross-fitted vanishing-point test + bend test -> per-image evidence

Output decision is three-valued. Without a calibrated empirical null (fitted
on held-out authentic images, the real-media null gate) the decision is left as "consistent"
or "inconsistent" by the chi^2 diagnostic only and is marked uncalibrated;
for any reported result an EmpiricalNull must be supplied.

Scope: natural scenes without straight, parallel
structure give no constraints and the branch ABSTAINS ("insufficient").
Coverage must be reported beside every accuracy number.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy import stats

from forensics.io.records import ConstraintEvidence, ImageEvidence
from forensics.stats.empirical_null import EmpiricalNull
from forensics.structure.collinearity import bend_test
from forensics.structure.lines import detect_lsd, segments_from_endpoints
from forensics.structure.vp_crossfit import cross_fitted_vp_test


def read_gray(path: Path | str) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def image_statistic(z_scores: np.ndarray) -> float:
    """Per-image statistic used for calibration: T / dof = mean of z^2."""
    return float(np.mean(np.square(z_scores))) if len(z_scores) else float("nan")


def analyze_gray(gray: np.ndarray, config: dict, rng: np.random.Generator, source: str = "",
                 empirical_null: EmpiricalNull | None = None, alpha: float = 0.05) -> ImageEvidence:
    height, width = gray.shape[:2]
    endpoints = detect_lsd(gray, config)
    segments = segments_from_endpoints(endpoints, width, height, config)
    records = []
    if len(segments) >= 2 * config["vanishing_points"]["min_fit_lines"]:
        records += cross_fitted_vp_test(segments, config, rng)
    records += bend_test(gray, endpoints, config)

    constraints = [
        ConstraintEvidence(kind=r["kind"], residual=r["residual"], z=r["z"], statistic=r["z"] ** 2, dof=1,
                           p_value_chi2=float(stats.chi2.sf(r["z"] ** 2, 1)),
                           extra={"length_px": r.get("length_px")})
        for r in records
    ]
    z = np.array([c.z for c in constraints])
    if len(z) < config["vanishing_points"]["min_test_lines"]:
        decision, p_value = "insufficient", float("nan")
    else:
        p_value = float(stats.chi2.sf(np.sum(z ** 2), len(z)))
        if empirical_null is not None:
            p_value = float(empirical_null.p_value(image_statistic(z))[0])
        decision = "inconsistent" if p_value <= alpha else "consistent"
    return ImageEvidence(
        source=source, width=width, height=height, n_segments=len(segments), n_constraints=len(z),
        decision=decision, statistic_sum=float(np.sum(z ** 2)) if len(z) else float("nan"), dof=len(z),
        p_value_chi2=p_value if empirical_null is None else float(stats.chi2.sf(np.sum(z ** 2), len(z))),
        median_abs_z=float(np.median(np.abs(z))) if len(z) else float("nan"),
        frac_abs_z_gt3=float(np.mean(np.abs(z) > 3)) if len(z) else float("nan"),
        constraints=constraints,
    )


def analyze_image(path: Path | str, config: dict, seed: int, empirical_null: EmpiricalNull | None = None,
                  min_short_side: bool = True) -> ImageEvidence:
    gray = read_gray(path)
    if min_short_side and min(gray.shape[:2]) < config["ingest"]["min_short_side_px"]:
        h, w = gray.shape[:2]
        return ImageEvidence(source=str(path), width=w, height=h, n_segments=0, n_constraints=0,
                             decision="insufficient", statistic_sum=float("nan"), dof=0,
                             p_value_chi2=float("nan"), median_abs_z=float("nan"), frac_abs_z_gt3=float("nan"))
    return analyze_gray(gray, config, np.random.default_rng(seed), source=str(path), empirical_null=empirical_null)
