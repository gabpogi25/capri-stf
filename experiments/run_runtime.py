"""
experiments/run_runtime.py

Runtime of the Python reference vs the C++ port, per stage.

Protocol
--------
* The wrapper functions dispatch to C++ whenever it is built, so Python
  timings run with CAPRI_BACKEND=python, and C++ timings call capri_cpp
  directly.
* Without C++ only Python rows are written; no C++ time is estimated.
* Both backends compute the same quantity in the syzygy stage: the max over
  the 15 Plucker relations from the points.
* Before any timing is trusted, each stage checks that the two backends give
  the same answer (relative difference <= 1e-10, same stratum). Disagreements are
  written to the CSV (column Outputs_Agree) instead of being hidden.

Point counts n in {12, 50, 100, 500}; 50 repetitions (10 for greedy); median and
IQR in milliseconds. Seed 42012.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import get_experiment_seed, make_rng, save_results, random_six_tuple
from algorithms.backend import get_cpp_module
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.syzygy.grassmann_plucker import compute_plucker_residual
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.degeneracy.stratification import classify_degeneracy_stratum
from algorithms.selection.greedy import select_tuples_greedy
from synthetic_data.generator import generate_generic_configuration

EXPERIMENT_ID = 12
SEED = get_experiment_seed(EXPERIMENT_ID)  # 42012
N_REPS = 50
N_REPS_GREEDY = 10
N_VALUES = [12, 50, 100, 500]


def time_call(func, n_reps: int) -> tuple[float, float, object]:
    """(median ms, IQR ms, last result) over n_reps calls."""
    timings, result = [], None
    for _ in range(n_reps):
        start = time.perf_counter()
        result = func()
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings)), float(np.percentile(timings, 75) - np.percentile(timings, 25)), result


def _as_values(result):
    """Normalize Python/C++ results (dict, tuple, InvariantResult, optional) to a float array."""
    if result is not None and hasattr(result, "j1"):
        return np.array([result.j1, result.j2, result.j3], dtype=float)
    if isinstance(result, dict):
        return np.asarray(list(result.values()), dtype=float).ravel()
    return np.asarray(result, dtype=float).ravel()


def _agree(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if hasattr(a, "name") or hasattr(b, "name"):
        return getattr(a, "name", a) == getattr(b, "name", b)
    a_arr, b_arr = _as_values(a), _as_values(b)
    return a_arr.shape == b_arr.shape and bool(
        np.all(np.abs(a_arr - b_arr) <= 1e-10 * np.maximum(1.0, np.abs(a_arr))))


def run_experiment() -> pd.DataFrame:
    print("=" * 60)
    print("EXP 12: RUNTIME, PYTHON REFERENCE VS C++ PORT")
    print("=" * 60)
    capri_cpp = get_cpp_module()
    print(f"  C++ module: {'available' if capri_cpp is not None else 'NOT available (Python rows only)'}")
    rng = make_rng(SEED)
    rows = []
    previous_backend = os.environ.get("CAPRI_BACKEND")

    for n in N_VALUES:
        pts = generate_generic_configuration(n, scale=1.0, rng=rng)
        t = tuple(int(i) for i in random_six_tuple(n, rng))
        stages = [
            ("Bracket Computation", lambda: compute_all_brackets_for_tuple(pts, t),
             (lambda: capri_cpp.compute_all_brackets(pts, t)) if capri_cpp else None, N_REPS),
            ("Invariant Eval", lambda: evaluate_six_point_invariants(pts, t),
             (lambda: capri_cpp.evaluate_six_point_invariants(pts, t, 1e-10)) if capri_cpp else None, N_REPS),
            ("Conditioning", lambda: compute_conditioning_score(pts, t),
             (lambda: capri_cpp.compute_conditioning_score(pts, t)) if capri_cpp else None, N_REPS),
            ("Plucker Residual (15 relations)", lambda: compute_plucker_residual(pts, t),
             (lambda: capri_cpp.compute_plucker_residual(pts, t)) if capri_cpp else None, N_REPS),
            ("Stratification", lambda: classify_degeneracy_stratum(pts, t),
             (lambda: capri_cpp.classify_degeneracy_stratum(pts, t, 1e-6, 1e-2)) if capri_cpp else None, N_REPS),
            ("Greedy Selection (k=20)",
             lambda: select_tuples_greedy(pts, conditioning_threshold=1000.0, max_tuples=20,
                                          rng=make_rng(SEED)),
             (lambda: capri_cpp.select_tuples_greedy(pts, 20, 1000.0, 1.0, 0.001, 10000, SEED))
             if capri_cpp else None, N_REPS_GREEDY),
        ]
        for stage, python_fn, cpp_fn, reps in stages:
            os.environ["CAPRI_BACKEND"] = "python"
            py_median, py_iqr, py_result = time_call(python_fn, reps)
            if previous_backend is None:
                os.environ.pop("CAPRI_BACKEND", None)
            else:
                os.environ["CAPRI_BACKEND"] = previous_backend
            agree = None
            if cpp_fn is not None and not stage.startswith("Greedy"):
                agree = _agree(py_result, cpp_fn())
            rows.append({"Points": n, "Stage": stage, "Language": "Python",
                         "Median_ms": py_median, "IQR_ms": py_iqr, "N_Reps": reps,
                         "Outputs_Agree": agree, "Seed": SEED})
            if cpp_fn is not None:
                cpp_median, cpp_iqr, _ = time_call(cpp_fn, reps)
                rows.append({"Points": n, "Stage": stage, "Language": "C++",
                             "Median_ms": cpp_median, "IQR_ms": cpp_iqr, "N_Reps": reps,
                             "Outputs_Agree": agree, "Seed": SEED})
            print(f"  n={n:4d} {stage:32s} Python {py_median:9.3f} ms"
                  + (f" | C++ {cpp_median:9.3f} ms | agree={agree}" if cpp_fn is not None else ""))

    df = pd.DataFrame(rows)
    print("\n  Greedy selection is not compared for equality: the two backends sample")
    print("  candidate tuples differently (see algorithms/selection/greedy.py).")
    save_results(df, f"runtime{os.environ.get('CAPRI_RESULTS_SUFFIX', '')}.csv", SEED, "runtime benchmark: Runtime, Python reference vs C++ port")
    return df


if __name__ == "__main__":
    run_experiment()
