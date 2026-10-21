"""
Math lock: the exact checks of docs/MATHEMATICS.md must keep passing.

Runs the exact / randomized-exact parts of experiments/symbolic/verify_pencil.py
(V1-V4) and of experiments/symbolic/verify_math_exact.py (W1-W5, on small n). The
mathematics is versioned in docs/MATHEMATICS.md (Section 10). Any change to the bracket, pencil or
gradient code that breaks a proved identity fails this test. The slow
numerical checks (V5-V8) run with `python experiments/symbolic/verify_pencil.py`.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "symbolic"))
sys.path.insert(0, str(ROOT))

import verify_math_exact as exact  # noqa: E402
import verify_pencil as vp  # noqa: E402


def _run(check, *args, **kwargs):
    vp.RESULTS.clear()
    check(np.random.default_rng(20261011), *args, **kwargs)
    failed = [name for name, ok in vp.RESULTS if not ok]
    assert vp.RESULTS and not failed, failed


def test_lemma_2_1_gauge_invariance():
    _run(vp.v1_gauge_invariance)


def test_prop_2_3_j_are_pencil_cross_ratios():
    _run(vp.v2_j_are_pencil_ratios)


def test_thm_2_5_plucker_is_cross_ratio_identity():
    _run(vp.v3_plucker_is_cross_ratio_relation)


def test_thm_3_2_generic_rank_n12():
    _run(vp.v4_rank)


def test_thm_3_2_generic_rank_n6():
    _run(vp.v4_rank, n=6, n_rows=45)


def _run_exact(check, *args, **kwargs):
    vp.RESULTS.clear()
    check(*args, **kwargs)
    failed = [name for name, ok in vp.RESULTS if not ok]
    assert vp.RESULTS and not failed, failed


def test_prop_3_5_frame_identities():
    _run_exact(exact.w1_frame_identities)


def test_prop_3_5_general_frame():
    _run_exact(exact.w2_general_frame, np.random.default_rng(20261018), n=7, trials=2)


def test_thm_3_7_rank_3n_minus_15():
    _run_exact(exact.w3_rank, np.random.default_rng(20261018), n_values=(6, 7, 9))


def test_lemma_3_6_stabiliser():
    _run_exact(exact.w4_stabiliser, np.random.default_rng(20261018))


def test_remark_3_9_six_point_closed_forms():
    _run_exact(exact.w5_six_point_formulas, np.random.default_rng(20261018))
