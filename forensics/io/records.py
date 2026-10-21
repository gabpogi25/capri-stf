"""
forensics/io/records.py

Typed records passed between pipeline stages.

Decisions are three-valued: "inconsistent", "consistent" or "insufficient".
The algebraic branch never forces a real/fake label; when the media has too
little usable geometry it abstains and says so.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

Decision = Literal["inconsistent", "consistent", "insufficient"]


@dataclass
class ConstraintEvidence:
    """One tested constraint F(x) = 0 (a line against a vanishing point, a bend test, a tuple)."""
    kind: str                       # "vp_angle", "bend", "delta_j"
    residual: float                 # r = F(x_hat) (or a vector summarized as its norm)
    z: float                        # residual standardized by its predicted standard deviation
    statistic: float                # T = r^T Sigma^+ r
    dof: int                        # degrees of freedom of T
    p_value_chi2: float             # 1 - F_chi2(T; dof), diagnostic only
    conditioning: float = float("nan")
    gate_passed: bool = True
    extra: dict = field(default_factory=dict)


@dataclass
class ImageEvidence:
    """Per-image (or per-frame) evidence from the single-image branch."""
    source: str
    width: int
    height: int
    n_segments: int
    n_constraints: int
    decision: Decision
    statistic_sum: float            # sum of z^2 over constraints
    dof: int
    p_value_chi2: float
    median_abs_z: float
    frac_abs_z_gt3: float
    constraints: list[ConstraintEvidence] = field(default_factory=list)

    def summary(self) -> dict:
        record = asdict(self)
        record.pop("constraints")
        return record


@dataclass
class ClipEvidence:
    """Per-clip evidence from the video branch."""
    source: str
    n_frames_sampled: int
    n_windows_total: int
    n_windows_used: int
    frame_p_values: list[float]
    window_statistics: list[float]
    clip_statistic: float
    clip_p_value_chi2: float
    effective_sample_size: float
    cusum_alarm_index: int | None
    temporal_tv: float
    decision: Decision
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        record = asdict(self)
        record["frame_p_values"] = [float(x) for x in self.frame_p_values]
        record["window_statistics"] = [float(x) for x in self.window_statistics]
        return record


def to_jsonable(value):
    """Convert numpy scalars/arrays inside nested records to plain Python types."""
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
