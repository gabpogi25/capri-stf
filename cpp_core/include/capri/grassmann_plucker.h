#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / grassmann_plucker.h
//
// Grassmann-Plucker relations for six-point bracket tables in P^3.
//
// This is the C++ equivalent of:
//   algorithms/syzygy/grassmann_plucker.py
// ─────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace capri {

struct PluckerRelation {
    std::pair<int, int> fixed_pair;
    std::array<int, 4> free_quad;
    double residual;
    std::array<double, 3> terms;
};

/// Evaluate all 15 Grassmann-Plucker relations for a six-point tuple.
/// @param points         (4, N) matrix.
/// @param tuple_indices  Array of 6 column indices.
/// @return Vector of 15 relation diagnostics.
std::vector<PluckerRelation> compute_all_plucker_relations(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices);

/// Compute the maximum absolute Plucker residual across all 15 relations.
/// @param points         (4, N) matrix.
/// @param tuple_indices  Array of 6 column indices.
/// @return max |residual| over all 15 relations (float >= 0).
double compute_plucker_residual(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices);

/// Compute the max Plucker residual for each tuple in a list.
std::vector<double> batch_plucker_residuals(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples);

}  // namespace capri
