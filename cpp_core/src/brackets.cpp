// ─────────────────────────────────────────────────────────────
// capri_cpp / brackets.cpp
//
// TODO: Port from algorithms/invariants/brackets.py
// ─────────────────────────────────────────────────────────────

#include "capri/brackets.h"
#include <algorithm>

namespace capri {

// Precomputed C(6,4) = 15 local quad index combinations
static constexpr std::array<std::array<int, 4>, 15> QUAD_COMBOS = {{
    {0,1,2,3}, {0,1,2,4}, {0,1,2,5}, {0,1,3,4}, {0,1,3,5},
    {0,1,4,5}, {0,2,3,4}, {0,2,3,5}, {0,2,4,5}, {0,3,4,5},
    {1,2,3,4}, {1,2,3,5}, {1,2,4,5}, {1,3,4,5}, {2,3,4,5},
}};

double compute_bracket(
    const Eigen::MatrixXd& points,
    int i, int j, int k, int l)
{
    Eigen::Matrix4d M;
    M.col(0) = points.col(i);
    M.col(1) = points.col(j);
    M.col(2) = points.col(k);
    M.col(3) = points.col(l);
    return M.determinant();
}

std::array<double, 15> compute_all_brackets(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& t)
{
    // Normalize to unit sphere
    Eigen::MatrixXd normed = points;
    for (int c = 0; c < normed.cols(); ++c) {
        normed.col(c).normalize();
    }

    std::array<double, 15> result;
    for (int q = 0; q < 15; ++q) {
        const auto& quad = QUAD_COMBOS[q];
        result[q] = compute_bracket(normed,
            t[quad[0]], t[quad[1]], t[quad[2]], t[quad[3]]);
    }
    return result;
}

std::vector<std::array<double, 15>> compute_all_brackets_batched(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples)
{
    // Normalize once
    Eigen::MatrixXd normed = points;
    for (int c = 0; c < normed.cols(); ++c) {
        normed.col(c).normalize();
    }

    std::vector<std::array<double, 15>> results(all_tuples.size());
    for (size_t k = 0; k < all_tuples.size(); ++k) {
        const auto& t = all_tuples[k];
        for (int q = 0; q < 15; ++q) {
            const auto& quad = QUAD_COMBOS[q];
            results[k][q] = compute_bracket(normed,
                t[quad[0]], t[quad[1]], t[quad[2]], t[quad[3]]);
        }
    }
    return results;
}

}  // namespace capri
