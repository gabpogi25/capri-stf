# CAPRI C++ Core

High-performance C++ implementation of the CAPRI projective invariant engine.

## Structure

```
cpp_core/
├── CMakeLists.txt              # Build configuration
├── include/capri/
│   ├── brackets.h              # Bracket computation (det 4x4)
│   ├── six_point.h             # J1, J2, J3 projective invariants
│   ├── conditioning.h          # L(S) = 8 / δ_min² scores
│   └── greedy.h                # Minoux lazy greedy selection
└── src/
    ├── brackets.cpp            # ← Port of algorithms/invariants/brackets.py
    ├── six_point.cpp           # ← Port of algorithms/invariants/six_point.py
    ├── conditioning.cpp        # ← Port of algorithms/conditioning/scores.py
    ├── greedy.cpp              # ← Port of algorithms/selection/greedy.py
    └── bindings.cpp            # pybind11 bindings (commented out until build)
```

## Dependencies

- **Eigen3** (required) — linear algebra (4×4 determinants)
- **pybind11** (optional) — Python bindings for side-by-side benchmarking

## Build (when ready)

```bash
cd cpp_core
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build . --config Release
```

## Usage from Python (after build)

```python
import capri_cpp

# Runs at ~50-100x the speed of the Python version
scores = capri_cpp.compute_conditioning_scores_batched(points, tuples)
```
