#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / residual.h
//
// Plücker consistency residual for six-point projective invariants.
//
// This is the C++ equivalent of:
//   algorithms/syzygy/residual.py
//
// Drift admissibility threshold (first-order bound):
//   tau(S) = 24 * delta_drift
// ─────────────────────────────────────────────────────────────

#include <array>
#include <cmath>

namespace capri {

/// Compute the Plücker consistency residual rho(S) from the 15 underlying brackets.
double compute_syzygy_residual(const std::array<double, 15>& brackets);

/// Compute the exact admissibility bound tau_exact(delta) = 3*eps*(2+eps)
/// where eps = (1+delta)^4 - 1.
double compute_exact_admissibility_bound(double delta_drift);

/// Return True if the consistency residual rho(S) <= tau_exact(delta_drift).
bool is_tuple_drift_admissible(
    const std::array<double, 15>& brackets,
    double delta_drift);

}  // namespace capri

