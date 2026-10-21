// ─────────────────────────────────────────────────────────────
// capri_cpp / greedy.cpp
//
// TODO: Port from algorithms/selection/greedy.py
// ─────────────────────────────────────────────────────────────

#include "capri/greedy.h"
#include "capri/conditioning.h"
#include <queue>
#include <random>
#include <set>
#include <algorithm>

namespace capri {

std::vector<std::array<int, 6>> select_tuples_greedy(
    const Eigen::MatrixXd& points,
    int max_tuples,
    double conditioning_threshold,
    double coverage_weight,
    double penalty_weight,
    int candidate_sample_size,
    unsigned int seed)
{
    const int n_points = static_cast<int>(points.cols());

    // Sample random candidate tuples
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n_points - 1);

    std::set<std::array<int, 6>> candidate_set;
    int consecutive_collisions = 0;
    while (static_cast<int>(candidate_set.size()) < candidate_sample_size && consecutive_collisions < 1000) {
        std::array<int, 6> t;
        std::set<int> seen;
        for (int i = 0; i < 6; ++i) {
            int v;
            do { v = dist(rng); } while (seen.count(v));
            seen.insert(v);
            t[i] = v;
        }
        std::sort(t.begin(), t.end());
        if (!candidate_set.insert(t).second) {
            consecutive_collisions++;
        } else {
            consecutive_collisions = 0;
        }
    }

    // Pre-filter by conditioning score
    struct Candidate {
        std::array<int, 6> tuple;
        double l_score;
    };
    std::vector<Candidate> viable;
    for (const auto& t : candidate_set) {
        double l_s = compute_conditioning_score(points, t);
        if (l_s < conditioning_threshold) {
            viable.push_back({t, l_s});
        }
    }

    if (viable.empty()) return {};

    // Lazy greedy (Minoux 1978) via max-heap
    std::set<int> covered;

    auto marginal_gain = [&](const Candidate& c) -> double {
        int new_cov = 0;
        for (int idx : c.tuple) {
            if (!covered.count(idx)) ++new_cov;
        }
        return coverage_weight * new_cov - penalty_weight * c.l_score;
    };

    // Max-heap: (score, index into viable)
    using Entry = std::pair<double, int>;
    std::priority_queue<Entry> heap;
    for (int k = 0; k < static_cast<int>(viable.size()); ++k) {
        heap.push({marginal_gain(viable[k]), k});
    }

    std::vector<std::array<int, 6>> selected;
    for (int iter = 0; iter < max_tuples && !heap.empty(); ++iter) {
        while (!heap.empty()) {
            auto [stale_score, k] = heap.top();
            heap.pop();
            double current = marginal_gain(viable[k]);
            if (heap.empty() || current >= heap.top().first) {
                if (current > 0 || selected.empty()) {
                    selected.push_back(viable[k].tuple);
                    for (int idx : viable[k].tuple) covered.insert(idx);
                }
                break;
            } else {
                heap.push({current, k});
            }
        }
    }
    return selected;
}

}  // namespace capri
