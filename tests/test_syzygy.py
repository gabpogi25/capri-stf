"""
tests/test_syzygy.py

Unit tests for the exact multi-view consistency residual (Phase 1).
"""
import numpy as np
import pytest
from algorithms.syzygy.residual import (
    compute_syzygy_residual,
    compute_exact_admissibility_bound,
    is_tuple_drift_admissible,
)
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from synthetic_data.generator import generate_generic_configuration

class TestExactBounds:
    def test_exact_bound_zero(self):
        tau = compute_exact_admissibility_bound(0.0)
        assert tau == 0.0

    def test_exact_bound_scales(self):
        tau_small = compute_exact_admissibility_bound(0.01)
        tau_large = compute_exact_admissibility_bound(0.1)
        assert tau_large > tau_small
        
    def test_rho_admissible_on_generic(self):
        rng = np.random.default_rng(42)
        pts = generate_generic_configuration(6, rng=rng)
        brackets = compute_all_brackets_for_tuple(pts, (0,1,2,3,4,5))
        assert is_tuple_drift_admissible(brackets, delta_drift=0.01)