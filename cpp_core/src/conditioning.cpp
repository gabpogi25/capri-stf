// ─────────────────────────────────────────────────────────────
// capri_cpp / conditioning.cpp
//
// TODO: Port from algorithms/conditioning/scores.py
// ─────────────────────────────────────────────────────────────

#include "capri/conditioning.h"
#include "capri/brackets.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include "capri/six_point.h"

namespace capri {

double compute_delta_min(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices)
{
    auto brackets = compute_all_brackets(points, tuple_indices);
    double min_abs = std::numeric_limits<double>::infinity();
    for (double b : brackets) {
        double ab = std::abs(b);
        if (ab < min_abs) min_abs = ab;
    }
    return min_abs;
}

double compute_conditioning_score(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices)
{
    double delta_min = compute_delta_min(points, tuple_indices);
    if (delta_min < 1e-15) {
        return std::numeric_limits<double>::infinity();
    }
    // L(S) = 2 * C_br / delta_min^2, C_br = 4.0 after unit-sphere norm
    return 8.0 / (delta_min * delta_min);
}

std::vector<double> compute_conditioning_scores_batched(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples)
{
    std::vector<double> scores(all_tuples.size());
    for (size_t k = 0; k < all_tuples.size(); ++k) {
        scores[k] = compute_conditioning_score(points, all_tuples[k]);
    }
    return scores;
}


double compute_chart_free_conditioning(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices,
    double noise_sigma)
{
    auto base_val = evaluate_six_point_invariants(points, tuple_indices);
    if (!base_val.has_value()) return std::numeric_limits<double>::infinity();

    double base_log_j1 = std::log(std::abs(base_val->j1));
    double base_log_j2 = std::log(std::abs(base_val->j2));
    double base_log_j3 = std::log(std::abs(base_val->j3));

    double grad_norm_sq_j1 = 0;
    double grad_norm_sq_j2 = 0;
    double grad_norm_sq_j3 = 0;

    double h = 1e-7;
    Eigen::MatrixXd perturbed_points = points;

    for (int col_idx = 0; col_idx < 6; ++col_idx) {
        int col = tuple_indices[col_idx];
        for (int row = 0; row < 4; ++row) {
            double orig = perturbed_points(row, col);
            perturbed_points(row, col) = orig + h;
            
            auto plus_val = evaluate_six_point_invariants(perturbed_points, tuple_indices);
            if (!plus_val.has_value()) return std::numeric_limits<double>::infinity();
            
            double d_j1 = (std::log(std::abs(plus_val->j1)) - base_log_j1) / h;
            double d_j2 = (std::log(std::abs(plus_val->j2)) - base_log_j2) / h;
            double d_j3 = (std::log(std::abs(plus_val->j3)) - base_log_j3) / h;
            
            grad_norm_sq_j1 += d_j1 * d_j1;
            grad_norm_sq_j2 += d_j2 * d_j2;
            grad_norm_sq_j3 += d_j3 * d_j3;
            
            perturbed_points(row, col) = orig;
        }
    }

    double max_grad_norm = std::sqrt(std::max({grad_norm_sq_j1, grad_norm_sq_j2, grad_norm_sq_j3}));
    return noise_sigma * max_grad_norm;
}

}  // namespace capri
