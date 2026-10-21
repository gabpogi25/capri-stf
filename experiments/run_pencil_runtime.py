"""
experiments/run_pencil_runtime.py

Runtime of the computations of docs/MATHEMATICS.md (Sections 4-5)
and of their implementations.

Stages (each: median and IQR over N_REPS calls, in milliseconds):
  kappa_batch      kappa_J for T = 10,000 tuples on n = 50 points:
                   per-tuple Python loop, vectorized NumPy batch, C++ batch
  pencil_score     pencil drift score Q_0.1 |Delta log|CR|| for two
                   reconstructions of n points (exact family for n <= 12,
                   otherwise DEFAULT_MAX_PENCILS sampled pencils): NumPy
                   brackets vs C++ brackets
  selector         kappa_J-feasible max-coverage selection, n = 30 points,
                   10,000 candidates: per-tuple scoring vs batch scoring

Before a timing is reported the implementations are checked against each
other (column Outputs_Agree): kappa relative difference <= 1e-9 with the same
set of infinite values, identical pencil scores up to 1e-12 relative, and
identical selected tuples. Seed 42112.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import get_experiment_seed, make_rng, save_results
from algorithms.backend import get_cpp_module
from algorithms.conditioning.scores import (
    _batch_invariant_noise_gain_numpy, batch_invariant_noise_gain, compute_invariant_noise_gain,
)
from algorithms.consistency.pencil import DEFAULT_MAX_PENCILS, pencil_drift_score, pencil_family_size
from algorithms.selection.coverage import select_tuples_max_coverage

SEED = get_experiment_seed(112)  # 42112
N_REPS = 5
PENCIL_N_VALUES = [12, 30, 100, 300]


def time_call(func, n_reps: int = N_REPS) -> tuple[float, float, object]:
    timings, result = [], None
    for _ in range(n_reps):
        start = time.perf_counter()
        result = func()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings)), float(np.percentile(timings, 75) - np.percentile(timings, 25)), result


def with_backend(mode: str, func):
    previous = os.environ.get("CAPRI_BACKEND")
    os.environ["CAPRI_BACKEND"] = mode
    try:
        return func()
    finally:
        if previous is None:
            os.environ.pop("CAPRI_BACKEND")
        else:
            os.environ["CAPRI_BACKEND"] = previous


def kappa_agree(a: np.ndarray, b: np.ndarray) -> bool:
    finite = np.isfinite(a)
    return bool(np.array_equal(finite, np.isfinite(b)) and np.allclose(a[finite], b[finite], rtol=1e-9))


def run_experiment() -> pd.DataFrame:
    print("=" * 60)
    print("EXP 12b: RUNTIME OF THE PENCIL / KAPPA COMPUTATIONS")
    print("=" * 60)
    has_cpp = get_cpp_module() is not None
    rng = make_rng(SEED)
    rows = []

    # kappa_J batch
    P = rng.standard_normal((4, 50))
    T = np.array([np.sort(rng.choice(50, 6, replace=False)) for _ in range(10_000)], dtype=np.int64)
    loop_ms, loop_iqr, loop = time_call(
        lambda: with_backend("python", lambda: np.array([compute_invariant_noise_gain(P, tuple(t)) for t in T])), 1)
    numpy_ms, numpy_iqr, numpy_values = time_call(lambda: _batch_invariant_noise_gain_numpy(P, T))
    rows += [
        {"Stage": "kappa_batch", "N_Points": 50, "Work_Items": len(T), "Implementation": "python_loop",
         "Median_ms": loop_ms, "IQR_ms": loop_iqr, "Outputs_Agree": True},
        {"Stage": "kappa_batch", "N_Points": 50, "Work_Items": len(T), "Implementation": "numpy_batch",
         "Median_ms": numpy_ms, "IQR_ms": numpy_iqr, "Outputs_Agree": kappa_agree(loop, numpy_values)},
    ]
    if has_cpp:
        cpp_ms, cpp_iqr, cpp_values = time_call(lambda: batch_invariant_noise_gain(P, T))
        rows.append({"Stage": "kappa_batch", "N_Points": 50, "Work_Items": len(T), "Implementation": "cpp_batch",
                     "Median_ms": cpp_ms, "IQR_ms": cpp_iqr, "Outputs_Agree": kappa_agree(loop, cpp_values)})

    # pencil drift score
    for n in PENCIL_N_VALUES:
        A = rng.standard_normal((4, n))
        B = A + 1e-3 * rng.standard_normal((4, n))
        work = min(pencil_family_size(n), DEFAULT_MAX_PENCILS)      # cross-ratio values evaluated
        py_ms, py_iqr, py_score = time_call(lambda: with_backend("python", lambda: pencil_drift_score(A, B)))
        rows.append({"Stage": "pencil_score", "N_Points": n, "Work_Items": work, "Implementation": "numpy",
                     "Median_ms": py_ms, "IQR_ms": py_iqr, "Outputs_Agree": True})
        if has_cpp:
            cpp_ms, cpp_iqr, cpp_score = time_call(lambda: pencil_drift_score(A, B))
            rows.append({"Stage": "pencil_score", "N_Points": n, "Work_Items": work, "Implementation": "cpp",
                         "Median_ms": cpp_ms, "IQR_ms": cpp_iqr,
                         "Outputs_Agree": bool(np.isclose(py_score, cpp_score, rtol=1e-12))})

    # max-coverage selector
    P30 = rng.standard_normal((4, 30))
    threshold = 50.0

    def select(batch: bool):
        return select_tuples_max_coverage(P30, compute_invariant_noise_gain, threshold, max_tuples=20,
                                          rng=make_rng(SEED + 1), max_candidates=10_000,
                                          batch_score_fn=batch_invariant_noise_gain if batch else None)

    single_ms, single_iqr, (single_sel, _) = time_call(lambda: select(False), 1)
    batch_ms, batch_iqr, (batch_sel, _) = time_call(lambda: select(True))
    rows += [
        {"Stage": "selector", "N_Points": 30, "Work_Items": 10_000, "Implementation": "per_tuple_scoring",
         "Median_ms": single_ms, "IQR_ms": single_iqr, "Outputs_Agree": True},
        {"Stage": "selector", "N_Points": 30, "Work_Items": 10_000, "Implementation": "batch_scoring",
         "Median_ms": batch_ms, "IQR_ms": batch_iqr, "Outputs_Agree": single_sel == batch_sel},
    ]

    df = pd.DataFrame(rows)
    df["Seed"] = SEED
    print(df.to_string(index=False))
    save_results(df, f"pencil_runtime{os.environ.get('CAPRI_RESULTS_SUFFIX', '')}.csv", SEED, "pencil runtime benchmark: runtime of the pencil and kappa_J computations")
    return df


if __name__ == "__main__":
    run_experiment()
