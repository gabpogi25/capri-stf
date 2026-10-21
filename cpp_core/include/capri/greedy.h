#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / greedy.h
//
// Minoux lazy greedy submodular tuple selection.
//
// This is the C++ equivalent of:
//   algorithms/selection/greedy.py
// ─────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace capri {

/// Select well-conditioned 6-point tuples via lazy greedy.
/// @param points                  (4, N) matrix.
/// @param max_tuples              Maximum tuples to select.
/// @param conditioning_threshold  L(S) rejection ceiling.
/// @param coverage_weight         Weight on new-point coverage.
/// @param penalty_weight          Weight on conditioning penalty.
/// @param candidate_sample_size   Random candidates to draw before greedy.
/// @param seed                    RNG seed.
/// @return Selected tuples.
std::vector<std::array<int, 6>> select_tuples_greedy(
    const Eigen::MatrixXd& points,
    int max_tuples = 500,
    double conditioning_threshold = 1000.0,
    double coverage_weight = 1.0,
    double penalty_weight = 0.001,
    int candidate_sample_size = 10000,
    unsigned int seed = 0);

}  // namespace capri
