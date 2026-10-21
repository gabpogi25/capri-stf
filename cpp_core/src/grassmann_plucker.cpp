// ─────────────────────────────────────────────────────────────
// capri_cpp / grassmann_plucker.cpp
//
// Port from algorithms/syzygy/grassmann_plucker.py
// ─────────────────────────────────────────────────────────────

#include "capri/grassmann_plucker.h"
#include "capri/brackets.h"
#include <cmath>
#include <algorithm>

namespace capri {

// Helper to map a sorted 4-tuple of local indices [0..5] to the bracket index [0..14]
static int get_bracket_index(std::array<int, 4> quad) {
    std::sort(quad.begin(), quad.end());
    // Lexicographic C(6,4) map
    if (quad == std::array<int,4>{0,1,2,3}) return 0;
    if (quad == std::array<int,4>{0,1,2,4}) return 1;
    if (quad == std::array<int,4>{0,1,2,5}) return 2;
    if (quad == std::array<int,4>{0,1,3,4}) return 3;
    if (quad == std::array<int,4>{0,1,3,5}) return 4;
    if (quad == std::array<int,4>{0,1,4,5}) return 5;
    if (quad == std::array<int,4>{0,2,3,4}) return 6;
    if (quad == std::array<int,4>{0,2,3,5}) return 7;
    if (quad == std::array<int,4>{0,2,4,5}) return 8;
    if (quad == std::array<int,4>{0,3,4,5}) return 9;
    if (quad == std::array<int,4>{1,2,3,4}) return 10;
    if (quad == std::array<int,4>{1,2,3,5}) return 11;
    if (quad == std::array<int,4>{1,2,4,5}) return 12;
    if (quad == std::array<int,4>{1,3,4,5}) return 13;
    if (quad == std::array<int,4>{2,3,4,5}) return 14;
    return -1;
}

// Helper to get bracket value with sign correction if indices are unsorted
static double get_bracket_val(const std::array<double, 15>& B, int i, int j, int k, int l) {
    std::array<int, 4> q = {i, j, k, l};
    int inversions = 0;
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            if (q[a] > q[b]) inversions++;
        }
    }
    int idx = get_bracket_index(q);
    double val = B[idx];
    return (inversions % 2 != 0) ? -val : val;
}

std::vector<PluckerRelation> compute_all_plucker_relations(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices)
{
    auto B = compute_all_brackets(points, tuple_indices);
    std::vector<PluckerRelation> results;
    results.reserve(15);

    for (int a = 0; a < 6; ++a) {
        for (int b = a + 1; b < 6; ++b) {
            std::vector<int> rem;
            for (int i = 0; i < 6; ++i) {
                if (i != a && i != b) rem.push_back(i);
            }
            int c = rem[0], d = rem[1], e = rem[2], f = rem[3];

            double t1 = get_bracket_val(B, a, b, c, d) * get_bracket_val(B, a, b, e, f);
            double t2 = get_bracket_val(B, a, b, c, e) * get_bracket_val(B, a, b, d, f);
            double t3 = get_bracket_val(B, a, b, c, f) * get_bracket_val(B, a, b, d, e);

            double residual = t1 - t2 + t3;

            results.push_back({
                {tuple_indices[a], tuple_indices[b]},
                {tuple_indices[c], tuple_indices[d], tuple_indices[e], tuple_indices[f]},
                residual,
                {t1, t2, t3}
            });
        }
    }
    return results;
}

double compute_plucker_residual(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& tuple_indices)
{
    auto relations = compute_all_plucker_relations(points, tuple_indices);
    double max_res = 0.0;
    for (const auto& r : relations) {
        if (std::abs(r.residual) > max_res) {
            max_res = std::abs(r.residual);
        }
    }
    return max_res;
}

std::vector<double> batch_plucker_residuals(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples)
{
    std::vector<double> results(all_tuples.size());
    for (size_t k = 0; k < all_tuples.size(); ++k) {
        results[k] = compute_plucker_residual(points, all_tuples[k]);
    }
    return results;
}

}  // namespace capri
