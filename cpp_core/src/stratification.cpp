// ─────────────────────────────────────────────────────────────
// capri_cpp / stratification.cpp
//
// Port from algorithms/degeneracy/stratification.py
// ─────────────────────────────────────────────────────────────

#include "capri/stratification.h"
#include "capri/brackets.h"
#include <cmath>
#include <array>

namespace capri {

std::string to_string(DegeneracyStratum s) {
    switch (s) {
        case DegeneracyStratum::GENERIC: return "generic";
        case DegeneracyStratum::BOUNDARY: return "boundary";
        case DegeneracyStratum::COMPOUND: return "compound";
        case DegeneracyStratum::CRITICAL: return "critical";
    }
    return "unknown";
}

DegeneracyStratum classify_degeneracy_stratum(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices,
    double near_zero_threshold,
    double boundary_threshold)
{
    auto B = compute_all_brackets(points, tuple_indices);

    // The denominator brackets in lexicographic order are at indices:
    // 1: (0124), 4: (0135), 7: (0235), 10: (1234), 8: (0245), 13: (1345)
    std::array<int, 6> denom_indices = {1, 4, 7, 10, 8, 13};
    for (int idx : denom_indices) {
        if (std::abs(B[idx]) < near_zero_threshold) {
            return DegeneracyStratum::CRITICAL;
        }
    }

    int near_zero_count = 0;
    for (int i = 0; i < 15; ++i) {
        if (std::abs(B[i]) < boundary_threshold) {
            near_zero_count++;
        }
    }

    if (near_zero_count == 0) {
        return DegeneracyStratum::GENERIC;
    } else if (near_zero_count == 1) {
        return DegeneracyStratum::BOUNDARY;
    } else {
        return DegeneracyStratum::COMPOUND;
    }
}

std::vector<DegeneracyStratum> batch_classify_strata(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples,
    double near_zero_threshold,
    double boundary_threshold)
{
    std::vector<DegeneracyStratum> results(all_tuples.size());
    for (size_t k = 0; k < all_tuples.size(); ++k) {
        results[k] = classify_degeneracy_stratum(
            points, all_tuples[k], near_zero_threshold, boundary_threshold);
    }
    return results;
}

}  // namespace capri
