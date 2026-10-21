#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / brackets.h
//
// GPU-ready bracket computation for projective point tuples in P^3.
// A bracket [p1 p2 p3 p4] = det(4x4 matrix of homogeneous columns).
//
// This is the C++ equivalent of:
//   algorithms/invariants/brackets.py
//   algorithms/invariants/brackets_torch.py
// ─────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <array>
#include <vector>

namespace capri {

/// Compute a single bracket [pi pj pk pl] = det of the 4x4 matrix.
/// Points must be unit-sphere normalized before calling.
/// @param points  (4, N) column-major matrix of homogeneous coordinates.
/// @param i,j,k,l Column indices into `points`.
/// @return Scalar bracket value.
double compute_bracket(
    const Eigen::MatrixXd& points,
    int i, int j, int k, int l);

/// All 15 brackets for a 6-point tuple, returned in lexicographic
/// order of C(6,4) local index combinations.
/// @param points         (4, N) matrix.
/// @param tuple_indices  Array of 6 column indices.
/// @return 15-element array of bracket values.
std::array<double, 15> compute_all_brackets(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices);

/// Batched: compute all 15 brackets for K tuples.
/// @param points          (4, N) matrix.
/// @param all_tuples      Vector of K tuples (each is 6 indices).
/// @return (K x 15) row-major vector of bracket values.
std::vector<std::array<double, 15>> compute_all_brackets_batched(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples);

}  // namespace capri
