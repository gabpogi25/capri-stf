"""
reconstruction/certification.py

Reliability certificate assembly for reconstructed 3D points.

Each reconstructed point receives a reliability score R(p) in [0, 1]
summarizing the algebraic quality of the correspondences that contributed
to its position. R(p) is not a single ML output; it is derived from:

  - ell(p): normalized conditioning signal, 1 - min(score / threshold, 1), for
            the active conditioning score (kappa_J by default, or L(S))
  - d(p):   drift consistency signal from rho vs tau margin, ONLY when rho is
            defined, i.e. when brackets come from different measurements.
            Within one reconstruction rho = 0 identically (docs/MATHEMATICS.md,
            Corollary 2.6); the pipeline then stores NaN and d(p) is dropped,
            with the remaining weights renormalized.
  - the descriptor confidence of the matches.

R(p) is a heuristic weighted mean, not a calibrated probability (trust-calibration experiment finds
ECE 0.21); calibrate it before reading it as a probability.

Optional calibrated mapping (calibrated-trust experiment, docs/prereg/trust_calibrated.md)
------------------------------------------------------------------------
Why a fitted map and not a closed form: R(p) is a weighted mean of clipped
signals. The probability that a point with a given R(p) is accurate depends on
the scene distribution (noise level, mismatch rate, planar fraction), which no
formula in R(p) encodes, so the only way to read R(p) as a probability is a
monotone map fitted to outcomes. TrustCalibration holds such a map (isotonic
knots, read from experiments/results/trust_calibrated_calibration_map.csv; no knot is
typed into code). Its output, PointCertificate.calibrated_trust, is
"marginally calibrated on the synthetic Section 4 distribution" only:
it estimates P(Y = 1 | R0(p), p in D, descriptor confidence 1, scene
distribution of Section 4 of the pre-registration), where Y = 1 means that the
median over the accepted tuples of p of max_k |ln|J_k(reconstruction)| -
ln|J_k(truth)|| is at most label_tolerance, R0 is the descriptor-free trust
score (compute_descriptor_free_trust_score) and D is the set of points in at
least one accepted tuple. It conditions on the pipeline's acceptance (kappa_J
gate at the recorded threshold and greedy coverage). Outside D the field is
None. It is not calibrated within strata (mismatched points, noise terciles)
or under distribution shift, and gives no guarantee on real images, other
thresholds or other descriptor models; the measured limits are in
experiments/results/trust_calibrated_strata.csv and trust_calibrated_metrics.csv. The default
outputs (trust_score, trust_label, export_certificates_to_dict) do not change
unless a TrustCalibration is passed.

Reference: GOAL.md Section 3 (Mathematical Accuracy Targets, drift derivation).
"""

from __future__ import annotations
import dataclasses
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field

TRUST_INPUT_CONVENTION = "descriptor_free_R0"
TRUST_DOMAIN = "|A(p)|>=1"
_MAP_METADATA_COLUMNS = ("input_convention", "conditioning_metric", "conditioning_threshold",
                         "label_tolerance", "method", "domain")


@dataclass
class TupleCertificate:
    """Certificate for a single six-point tuple."""
    tuple_indices: tuple
    conditioning_score: float
    consistency_residual: float
    drift_threshold: float
    stratum: str
    accepted: bool
    descriptor_confidence: float = 1.0


@dataclass
class PointCertificate:
    """Reliability certificate for a single reconstructed 3D point."""
    point_index: int
    position_homogeneous: np.ndarray        # (4,) homogeneous coords
    contributing_tuples: list[TupleCertificate] = field(default_factory=list)
    trust_score: float = 0.0               # R(p) in [0, 1]
    trust_label: str = "unassigned"        # "high", "moderate", "low"
    # P(Y = 1 | R0(p), p in D) under the synthetic Section 4 distribution
    # (marginally calibrated there only); None outside D or without a map.
    calibrated_trust: float | None = None


@dataclass(frozen=True)
class TrustCalibration:
    """
    Monotone map from the descriptor-free trust score R0(p) to a probability.

    The map is marginally calibrated on the synthetic Section 4 distribution
    of docs/prereg/trust_calibrated.md, for points in the item domain D
    (|A(p)| >= 1, accepted under conditioning_metric at conditioning_threshold)
    and the invariant-accuracy label at label_tolerance. It is not
    conditionally calibrated (strata) and not calibrated under shift.
    predict is piecewise-linear interpolation between the isotonic knots with
    clipping outside them, which is what sklearn's IsotonicRegression(
    out_of_bounds="clip").predict computes.
    """
    knots_r0: tuple[float, ...]
    knots_probability: tuple[float, ...]
    input_convention: str
    conditioning_metric: str
    conditioning_threshold: float
    label_tolerance: float
    method: str = "isotonic"
    source_experiment: str = "unknown"
    source_commit: str = "unknown"

    def __post_init__(self) -> None:
        x = np.asarray(self.knots_r0, float)
        y = np.asarray(self.knots_probability, float)
        if x.ndim != 1 or x.shape != y.shape or len(x) < 1:
            raise ValueError(f"knots must be two 1-D sequences of equal length, got {x.shape} and {y.shape}")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("knots must be finite")
        if np.any(np.diff(x) <= 0):
            raise ValueError("knot positions must increase strictly")
        if np.any(np.diff(y) < 0):
            raise ValueError("knot probabilities must not decrease")
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError(f"knot probabilities must lie in [0, 1], got [{y.min()}, {y.max()}]")
        if self.input_convention != TRUST_INPUT_CONVENTION:
            raise ValueError(f"input_convention {self.input_convention!r} != {TRUST_INPUT_CONVENTION!r}")

    def predict(self, descriptor_free_trust: np.ndarray | float) -> np.ndarray:
        """Calibrated probability for R0 values (monotone, clipped to the knot range)."""
        r0 = np.asarray(descriptor_free_trust, float)
        assert np.all(np.isfinite(r0)), "R0 must be finite"
        x = np.asarray(self.knots_r0, float)
        return np.interp(np.clip(r0, x[0], x[-1]), x, np.asarray(self.knots_probability, float))


def _read_provenance_header(path: Path) -> dict[str, str]:
    """Key-value pairs of the '# Key: value' header written by save_results."""
    header: dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            header[key.strip()] = value.strip()
    return header


def load_trust_calibration(
    path: str | Path,
    expected_threshold: float,
    expected_metric: str = "kappa",
) -> TrustCalibration:
    """
    Read trust_calibrated_calibration_map.csv (provenance header skipped) and check that
    it was fitted on the population it will be applied to.

    expected_threshold is required (certification.py cannot import the
    pipeline): pass the conditioning threshold the reliability layer uses.
    Raises ValueError on any mismatch, naming both values.
    """
    import pandas as pd

    path = Path(path)
    table = pd.read_csv(path, comment="#")
    for column in ("x", "y", *_MAP_METADATA_COLUMNS):
        if column not in table.columns:
            raise ValueError(f"{path.name}: missing column {column!r}")
        if column not in ("x", "y") and table[column].nunique(dropna=False) != 1:
            raise ValueError(f"{path.name}: column {column!r} holds more than one value")
    meta = {c: table[c].iloc[0] for c in _MAP_METADATA_COLUMNS}
    if expected_metric != "kappa" or meta["conditioning_metric"] != expected_metric:
        raise ValueError(f"conditioning_metric {meta['conditioning_metric']!r} != expected {expected_metric!r} "
                         "(only 'kappa' maps exist)")
    stored = float(meta["conditioning_threshold"])
    if not np.isclose(stored, expected_threshold, rtol=1e-9, atol=0.0):
        raise ValueError(f"conditioning_threshold {stored!r} != expected {expected_threshold!r}")
    if meta["domain"] != TRUST_DOMAIN:
        raise ValueError(f"domain {meta['domain']!r} != {TRUST_DOMAIN!r}")
    header = _read_provenance_header(path)
    return TrustCalibration(
        knots_r0=tuple(float(v) for v in table["x"]),
        knots_probability=tuple(float(v) for v in table["y"]),
        input_convention=str(meta["input_convention"]),
        conditioning_metric=str(meta["conditioning_metric"]),
        conditioning_threshold=stored,
        label_tolerance=float(meta["label_tolerance"]),
        method=str(meta["method"]),
        source_experiment=header.get("Experiment", "unknown"),
        source_commit=header.get("Code commit", "unknown"),
    )


def compute_conditioning_signal(
    conditioning_score: float,
    conditioning_threshold: float = 1000.0
) -> float:
    """
    Compute the normalized conditioning signal ell in [0, 1].

    ell = 1 - min(L(S) / L_threshold, 1)
    ell -> 1: excellent conditioning (L(S) << L_threshold)
    ell -> 0: marginal conditioning (L(S) -> L_threshold)

    Args:
        conditioning_score: L(S) for the tuple.
        conditioning_threshold: L_threshold.

    Returns:
        ell in [0, 1].
    """
    ratio = min(conditioning_score / max(conditioning_threshold, 1e-9), 1.0)
    return float(1.0 - ratio)


def compute_drift_signal(
    consistency_residual: float,
    drift_threshold: float
) -> float:
    """
    Compute the drift consistency signal d in [0, 1].

    d = 1 if rho = 0, d = 0 if rho >= tau(S).
    Linear interpolation between 0 and tau(S).

    Args:
        consistency_residual: rho for this tuple.
        drift_threshold: tau(S) for this tuple.

    Returns:
        d in [0, 1].
    """
    if drift_threshold <= 0:
        return float(consistency_residual < 1e-10)
    ratio = min(consistency_residual / drift_threshold, 1.0)
    return float(1.0 - ratio)


def compute_point_trust_score(
    certificates: list[TupleCertificate],
    conditioning_threshold: float = 1000.0,
    conditioning_weight: float = 0.5,
    drift_weight: float = 0.3,
    descriptor_weight: float = 0.2,
) -> float:
    """
    Compute R(p) for a point from its contributing tuple certificates.

    R(p) = mean over contributing accepted tuples of:
        conditioning_weight * ell(t) +
        drift_weight * d(t) +
        descriptor_weight * desc_conf(t)

    If no accepted tuples contribute to this point, R(p) = 0.

    Args:
        certificates: List of TupleCertificate for tuples involving this point.
        conditioning_threshold: L_threshold for normalization.
        conditioning_weight, drift_weight, descriptor_weight: Must sum to 1.

    Returns:
        R(p) in [0, 1].
    """
    assert abs(conditioning_weight + drift_weight + descriptor_weight - 1.0) < 1e-9, (
        "Trust score weights must sum to 1.0"
    )

    accepted = [c for c in certificates if c.accepted]
    if not accepted:
        return 0.0

    scores = []
    for cert in accepted:
        ell = compute_conditioning_signal(cert.conditioning_score, conditioning_threshold)
        desc = cert.descriptor_confidence
        if np.isfinite(cert.consistency_residual) and np.isfinite(cert.drift_threshold):
            d = compute_drift_signal(cert.consistency_residual, cert.drift_threshold)
            score = conditioning_weight * ell + drift_weight * d + descriptor_weight * desc
        else:
            # rho undefined (single reconstruction, Cor. 2.6): drop d and renormalize
            score = (conditioning_weight * ell + descriptor_weight * desc) / (conditioning_weight + descriptor_weight)
        scores.append(score)

    return float(np.mean(scores))


def compute_descriptor_free_trust_score(
    certificates: list[TupleCertificate],
    conditioning_threshold: float,
) -> float | None:
    """
    R0(p) = (0.5 * mean_{t in A(p)} ell(t) + 0.2) / 0.7, the input of TrustCalibration.

    This is compute_point_trust_score with every descriptor confidence set to 1
    and the drift term dropped (as within one reconstruction, Cor. 2.6), so it
    depends on the accepted certificates' conditioning scores only. Returns
    None when no certificate is accepted (p outside D).
    """
    assert conditioning_threshold > 0, "conditioning_threshold must be positive"
    accepted = [
        dataclasses.replace(c, descriptor_confidence=1.0, consistency_residual=float("nan"),
                            drift_threshold=float("nan"))
        for c in certificates if c.accepted
    ]
    if not accepted:
        return None
    return compute_point_trust_score(accepted, conditioning_threshold)


def compute_calibrated_trust(
    certificates: list[TupleCertificate],
    conditioning_threshold: float,
    trust_calibration: TrustCalibration,
) -> float | None:
    """
    calibrated_trust for one point: None outside D, else predict(R0(p)).

    Raises ValueError when an accepted certificate has a finite consistency
    residual and drift threshold (a drift term the map was never fitted with).
    """
    accepted = [c for c in certificates if c.accepted]
    if not accepted:
        return None
    for cert in accepted:
        if np.isfinite(cert.consistency_residual) and np.isfinite(cert.drift_threshold):
            raise ValueError(f"tuple {cert.tuple_indices} has a finite drift term "
                             f"(rho={cert.consistency_residual}, tau={cert.drift_threshold}); "
                             "the trust calibration was fitted without one")
    r0 = compute_descriptor_free_trust_score(accepted, conditioning_threshold)
    return float(trust_calibration.predict(r0))


def assign_trust_label(trust_score: float) -> str:
    """
    Map a trust score to a qualitative label for visualization.

    Args:
        trust_score: R(p) in [0, 1].

    Returns:
        "high" (R >= 0.7), "moderate" (0.4 <= R < 0.7), or "low" (R < 0.4).
    """
    if trust_score >= 0.7:
        return "high"
    elif trust_score >= 0.4:
        return "moderate"
    else:
        return "low"


def assemble_point_certificates(
    points_3d: np.ndarray,
    tuple_certificates: list[TupleCertificate],
    conditioning_threshold: float = 1000.0,
    trust_calibration: TrustCalibration | None = None,
) -> list[PointCertificate]:
    """
    Assemble per-point reliability certificates from tuple-level certificates.

    For each reconstructed 3D point, collects all tuples that include it
    and computes R(p) from those tuples' certificates.

    Args:
        points_3d: (4, M) homogeneous 3D point array.
        tuple_certificates: List of TupleCertificate for all evaluated tuples.
        conditioning_threshold: For normalization.
        trust_calibration: Optional map (load_trust_calibration). Without it the
            output is unchanged and calibrated_trust is None. With it,
            calibrated_trust = predict(R0(p)) for points in D, None outside D;
            ValueError if conditioning_threshold differs from the map's (relative
            tolerance 1e-9) or an accepted tuple carries a finite drift term.
            trust_score always keeps the pipeline's descriptor confidences.

    Returns:
        List of M PointCertificate objects.
    """
    if trust_calibration is not None and not np.isclose(
            conditioning_threshold, trust_calibration.conditioning_threshold, rtol=1e-9, atol=0.0):
        raise ValueError(f"conditioning_threshold {conditioning_threshold!r} != trust calibration's "
                         f"{trust_calibration.conditioning_threshold!r}")
    n_points = points_3d.shape[1]
    point_certs: list[PointCertificate] = []

    for p_idx in range(n_points):
        contributing = [
            c for c in tuple_certificates
            if p_idx in c.tuple_indices
        ]
        trust = compute_point_trust_score(contributing, conditioning_threshold)
        label = assign_trust_label(trust)
        calibrated = (None if trust_calibration is None else
                      compute_calibrated_trust(contributing, conditioning_threshold, trust_calibration))
        cert = PointCertificate(
            point_index=p_idx,
            position_homogeneous=points_3d[:, p_idx].copy(),
            contributing_tuples=contributing,
            trust_score=trust,
            trust_label=label,
            calibrated_trust=calibrated,
        )
        point_certs.append(cert)

    return point_certs


def export_certificates_to_dict(
    point_certs: list[PointCertificate],
    include_calibrated: bool = False,
) -> dict:
    """
    Export point certificates to a JSON-serializable dictionary for diagnostics.

    Args:
        point_certs: List of PointCertificate objects.
        include_calibrated: If True, add "calibrated_trust" per point (None,
            i.e. JSON null, outside D or without a map). Default output unchanged.

    Returns:
        Dictionary with per-point diagnostics.
    """
    points = []
    for c in point_certs:
        entry = {
            "index": c.point_index,
            "trust_score": round(c.trust_score, 6),
            "trust_label": c.trust_label,
            "n_contributing_tuples": len(c.contributing_tuples),
            "n_accepted_tuples": sum(1 for t in c.contributing_tuples if t.accepted),
        }
        if include_calibrated:
            entry["calibrated_trust"] = (None if c.calibrated_trust is None
                                         else round(float(c.calibrated_trust), 6))
        points.append(entry)
    return {"n_points": len(point_certs), "points": points}