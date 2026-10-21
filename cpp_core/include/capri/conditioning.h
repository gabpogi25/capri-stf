#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / conditioning.h
//
// Conditioning scores L(S) = 2 * C_br / delta_min^2.
//
// This is the C++ equivalent of:
//   algorithms/conditioning/scores.py
// ─────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace capri {

/// Compute delta_min(S) = min |bracket| over all C(6,4) sub-selections.
double compute_delta_min(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices);

/// Compute L(S) = 8.0 / delta_min^2 (after unit-sphere normalization).
double compute_conditioning_score(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices);

/// Batched conditioning scores for K tuples.
std::vector<double> compute_conditioning_scores_batched(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples);

/// Compute chart-free conditioning score \kappa_J(x) = ||\Sigma_0^{1/2} \nabla_x \log J_i||
double compute_chart_free_conditioning(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices,
    double noise_sigma = 1e-4);

}  // namespace capri
