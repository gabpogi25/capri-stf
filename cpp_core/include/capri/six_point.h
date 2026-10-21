#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / six_point.h
//
// Six-point projective invariants J1, J2, J3 for P^3.
//
// This is the C++ equivalent of:
//   algorithms/invariants/six_point.py
//   algorithms/invariants/six_point_torch.py
// ─────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <array>
#include <optional>
#include <vector>

namespace capri {

/// Result of invariant evaluation for a single tuple.
struct InvariantResult {
    double j1, j2, j3;
};

/// Compute (J1, J2, J3) for a single 6-point tuple.
/// Returns std::nullopt if any denominator bracket is below denom_tolerance.
/// @param points         (4, N) unit-sphere-normalized matrix.
/// @param tuple_indices  6-element array of column indices.
/// @param denom_tol      Minimum absolute denominator magnitude.
std::optional<InvariantResult> evaluate_six_point_invariants(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices,
    double denom_tol = 1e-10);

/// Batched evaluation for K tuples.
/// @param points      (4, N) matrix.
/// @param all_tuples  Vector of K tuples.
/// @param denom_tol   Degeneracy threshold.
/// @return Pair of: (K x 3 invariant values, K-element validity mask).
std::pair<std::vector<InvariantResult>, std::vector<bool>>
evaluate_batch_invariants(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples,
    double denom_tol = 1e-10);

}  // namespace capri
