"""
algorithms/backend.py

Explicit selection of the numerical backend for CAPRI.

Two implementations of the core bracket/invariant/conditioning/greedy code
exist: the pure-Python reference in algorithms/ and the C++ port in cpp_core/
(compiled with `make cpp`). Importing the C++ module silently whenever it
happens to be importable would let the same script produce different numbers
on different machines without saying so.

The backend is therefore chosen explicitly through the environment variable
CAPRI_BACKEND:

    auto    (default) use C++ if it has been built, otherwise Python
    python  always use the Python reference implementation
    cpp     require the C++ module; raise ImportError if it is missing

Every experiment records backend_name() in the metadata header of its CSV,
so each published number can be traced to the implementation that made it.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VALID_MODES = ("auto", "python", "cpp")
_module_cache: dict[str, ModuleType | None] = {}


def _requested_mode() -> str:
    mode = os.environ.get("CAPRI_BACKEND", "auto").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"CAPRI_BACKEND must be one of {_VALID_MODES}, got {mode!r}")
    return mode


def _load_cpp_module() -> ModuleType | None:
    if "capri_cpp" not in _module_cache:
        for candidate in (_PROJECT_ROOT / "cpp_core", _PROJECT_ROOT / "cpp_core" / "build"):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        try:
            import capri_cpp  # noqa: PLC0415 (optional compiled extension)
            _module_cache["capri_cpp"] = capri_cpp
        except ImportError:
            _module_cache["capri_cpp"] = None
    return _module_cache["capri_cpp"]


def get_cpp_module() -> ModuleType | None:
    """Return the compiled capri_cpp module, or None when Python is selected."""
    mode = _requested_mode()
    if mode == "python":
        return None
    module = _load_cpp_module()
    if mode == "cpp" and module is None:
        raise ImportError(
            "CAPRI_BACKEND=cpp but the capri_cpp extension is not built. "
            "Run `make cpp` (or `python cpp_core/setup.py build_ext --inplace` "
            "from inside cpp_core/)."
        )
    return module


def backend_name() -> str:
    """Name of the backend that get_cpp_module() resolves to: 'cpp' or 'python'."""
    return "cpp" if get_cpp_module() is not None else "python"
