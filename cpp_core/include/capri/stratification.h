#pragma once
// ─────────────────────────────────────────────────────────────
// capri_cpp / stratification.h
//
// Degeneracy stratum classification for six-point projective configurations.
//
// This is the C++ equivalent of:
//   algorithms/degeneracy/stratification.py
// ─────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <string>

namespace capri {

enum class DegeneracyStratum {
    GENERIC,
    BOUNDARY,
    COMPOUND,
    CRITICAL
};

/// Convert stratum enum to string for python/logging
std::string to_string(DegeneracyStratum s);

/// Classify the degeneracy stratum of a six-point tuple by bracket analysis.
/// @param points              (4, N) matrix.
/// @param tuple_indices       Array of 6 column indices.
/// @param near_zero_threshold Bracket magnitude for CRITICAL threshold.
/// @param boundary_threshold  Bracket magnitude for BOUNDARY threshold.
/// @return The stratum enum value.
DegeneracyStratum classify_degeneracy_stratum(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices,
    double near_zero_threshold = 1e-6,
    double boundary_threshold = 1e-2);

/// Batched version: classify strata for K tuples.
std::vector<DegeneracyStratum> batch_classify_strata(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples,
    double near_zero_threshold = 1e-6,
    double boundary_threshold = 1e-2);

}  // namespace capri
