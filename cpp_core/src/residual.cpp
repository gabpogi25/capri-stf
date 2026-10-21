// ─────────────────────────────────────────────────────────────
// capri_cpp / residual.cpp
//
// Port from algorithms/syzygy/residual.py
//
// Drift admissibility threshold (first-order bound):
//   tau(S) = 24 * delta_drift
// ─────────────────────────────────────────────────────────────

#include "capri/residual.h"
#include <algorithm>
#include <stdexcept>
#include <array>

namespace capri {

double compute_syzygy_residual(const std::array<double, 15>& brackets)
{
    // Sigma 1: [0123][0145] - [0124][0135] + [0125][0134]
    double sigma_1 = brackets[0] * brackets[5] 
                   - brackets[1] * brackets[4] 
                   + brackets[2] * brackets[3];

    // Sigma 2: [0234][1235] - [0235][1234] - [0123][2345]
    double sigma_2 = brackets[6] * brackets[11] 
                   - brackets[7] * brackets[10] 
                   - brackets[0] * brackets[14];

    // Sigma 3: [0145][2345] - [0245][1345] + [0345][1245]
    double sigma_3 = brackets[5] * brackets[14] 
                   - brackets[8] * brackets[13] 
                   + brackets[9] * brackets[12];

    return std::max({std::abs(sigma_1), std::abs(sigma_2), std::abs(sigma_3)});
}

double compute_exact_admissibility_bound(double delta_drift)
{
    if (delta_drift < 0) {
        throw std::invalid_argument("Delta drift must be non-negative.");
    }
    double eps = std::pow(1.0 + delta_drift, 4) - 1.0;
    return 3.0 * eps * (2.0 + eps);
}

bool is_tuple_drift_admissible(
    const std::array<double, 15>& brackets,
    double delta_drift)
{
    double rho = compute_syzygy_residual(brackets);
    double tau = compute_exact_admissibility_bound(delta_drift);
    return rho <= tau;
}

}  // namespace capri

