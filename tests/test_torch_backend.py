"""The optional PyTorch backend agrees with the NumPy reference (skipped without PyTorch)."""

import numpy as np
import pytest

from algorithms.consistency import torch_backend
from algorithms.consistency.pencil import pencil_discrepancies, pencil_index

pytestmark = pytest.mark.skipif(not torch_backend.available(), reason="PyTorch not installed")


def test_brackets_and_discrepancies_match_numpy():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 12))
    B = A + 1e-3 * rng.normal(size=(4, 12))
    quads, terms = pencil_index(12)
    unit = A / np.linalg.norm(A, axis=0)
    ref = np.linalg.slogdet(np.transpose(unit[:, quads], (1, 0, 2)))[1]
    assert np.allclose(torch_backend.log_abs_brackets(A, quads, "cpu"), ref, atol=1e-9)
    D = torch_backend.pencil_discrepancies(A, B, quads, terms, "cpu")
    assert np.allclose(D, pencil_discrepancies(A, B), atol=1e-9)
    assert not np.allclose(D, pencil_discrepancies(A, A + 2e-3 * rng.normal(size=(4, 12))), atol=1e-9)  # control
