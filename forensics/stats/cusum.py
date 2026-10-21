"""
forensics/stats/cusum.py

One-sided CUSUM for the onset of geometric inconsistency in a clip:
    S_t = max(0, S_{t-1} + u_t - k),  alarm when S_t > h,
where u_t = Phi^{-1}(1 - p_t) is the per-frame evidence on the z scale.
h is tuned on authentic validation clips so that the average run length under
the null is at least the target (default 500 frames).
"""

from __future__ import annotations
import numpy as np


def cusum(u: np.ndarray, k: float = 0.5, h: float = 5.0) -> tuple[np.ndarray, int | None]:
    """CUSUM path and the index of the first alarm (None if no alarm)."""
    path = np.zeros(len(u))
    s = 0.0
    alarm = None
    for t, value in enumerate(np.asarray(u, dtype=float)):
        s = max(0.0, s + value - k)
        path[t] = s
        if alarm is None and s > h:
            alarm = t
    return path, alarm


def average_run_length(null_series: list[np.ndarray], k: float, h: float) -> float:
    """Mean first-alarm time over authentic clips (clip length when no alarm: a lower bound)."""
    run_lengths = []
    for series in null_series:
        _, alarm = cusum(series, k, h)
        run_lengths.append(len(series) if alarm is None else alarm + 1)
    return float(np.mean(run_lengths)) if run_lengths else float("nan")


def calibrate_h(null_series: list[np.ndarray], k: float = 0.5, target_arl: float = 500.0,
                candidates=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0)) -> float:
    """Smallest h in candidates whose empirical ARL_0 reaches the target (largest h otherwise)."""
    for h in sorted(candidates):
        if average_run_length(null_series, k, h) >= target_arl:
            return float(h)
    return float(max(candidates))


def temporal_total_variation(u: np.ndarray) -> float:
    """mean |u_t - u_{t-1}| (diagnostic of flicker-like instability)."""
    u = np.asarray(u, dtype=float)
    return float(np.mean(np.abs(np.diff(u)))) if u.size > 1 else 0.0
