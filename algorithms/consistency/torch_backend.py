"""
algorithms/consistency/torch_backend.py

Optional PyTorch implementation of the batch computations behind the pencil statistic
(plan item B2), for large scenes on a GPU. It is opt-in: nothing in the pipeline
or the experiments imports it, so every recorded result keeps its C++ / NumPy backend
(algorithms/backend.py). In float64 it agrees with the NumPy reference to 1e-9
(tests/test_torch_backend.py); GPU timings are to be measured on the author's machine.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:                                   # optional dependency
    torch = None


def available() -> bool:
    return torch is not None


def _device(device: str | None) -> "torch.device":
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def log_abs_brackets(points: np.ndarray, quads: np.ndarray, device: str | None = None) -> np.ndarray:
    """log|det| of the unit-normalised columns of each quadruple (same as pencil.log_abs_brackets)."""
    assert torch is not None, "PyTorch is not installed"
    dev = _device(device)
    P = torch.as_tensor(points, dtype=torch.float64, device=dev)
    U = P / torch.linalg.norm(P, dim=0, keepdim=True)
    Q = torch.as_tensor(np.asarray(quads, dtype=np.int64), device=dev)
    M = U[:, Q].permute(1, 0, 2)                                       # (K, 4, 4)
    return torch.linalg.slogdet(M).logabsdet.cpu().numpy()


def pencil_discrepancies(points_a: np.ndarray, points_b: np.ndarray, quads: np.ndarray, terms: np.ndarray,
                         device: str | None = None) -> np.ndarray:
    """|log|CR_t(A)| - log|CR_t(B)|| for the pencils given by bracket quads and (T, 4) term indices
    (as returned by pencil.pencil_index): log|CR_t| = L[t0] + L[t1] - L[t2] - L[t3]."""
    La, Lb = log_abs_brackets(points_a, quads, device), log_abs_brackets(points_b, quads, device)
    d = La - Lb
    t = np.asarray(terms)
    with np.errstate(invalid="ignore"):
        return np.abs(d[t[:, 0]] + d[t[:, 1]] - d[t[:, 2]] - d[t[:, 3]])
