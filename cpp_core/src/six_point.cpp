// ─────────────────────────────────────────────────────────────
// capri_cpp / six_point.cpp
//
// TODO: Port from algorithms/invariants/six_point.py
// ─────────────────────────────────────────────────────────────

#include "capri/six_point.h"
#include "capri/brackets.h"
#include <cmath>

namespace capri {

std::optional<InvariantResult> evaluate_six_point_invariants(
    const Eigen::MatrixXd& points,
    const std::array<int, 6>& t,
    double denom_tol)
{
    auto B = compute_all_brackets(points, t);

    // Map local quad -> bracket index in the 15-element array.
    // Bracket order follows lexicographic C(6,4):
    //  0:(0123) 1:(0124) 2:(0125) 3:(0134) 4:(0135)
    //  5:(0145) 6:(0234) 7:(0235) 8:(0245) 9:(0345)
    // 10:(1234) 11:(1235) 12:(1245) 13:(1345) 14:(2345)

    // J1 = [abcd][abef] / ([abce][abdf])
    //     = [0123][0145] / ([0124][0135])
    double j1_num = B[0] * B[5];   // [0123]*[0145]
    double j1_den = B[1] * B[4];   // [0124]*[0135]

    // J2 = [acde][bcdf] / ([acdf][bcde])
    //     = [0234][1235] / ([0235][1234])
    double j2_num = B[6] * B[11];  // [0234]*[1235]  -- WRONG: should be [0234]=[acde] but a=0,c=2,d=3,e=4 -> [0234]
    // Actually let's be precise with the mapping:
    // a=0, b=1, c=2, d=3, e=4, f=5
    // J2 = [acde][bcdf] / ([acdf][bcde])
    //    = [0,2,3,4][1,2,3,5] / ([0,2,3,5][1,2,3,4])
    j2_num = B[6] * B[11];  // [0234]*[1235]
    double j2_den = B[7] * B[10];  // [0235]*[1234]

    // J3 = [abef][cdef] / ([acef][bdef])
    //    = [0,1,4,5][2,3,4,5] / ([0,2,4,5][1,3,4,5])
    double j3_num = B[5] * B[14];  // [0145]*[2345]
    double j3_den = B[8] * B[13];  // [0245]*[1345]

    if (std::abs(j1_den) < denom_tol ||
        std::abs(j2_den) < denom_tol ||
        std::abs(j3_den) < denom_tol) {
        return std::nullopt;
    }

    return InvariantResult{
        j1_num / j1_den,
        j2_num / j2_den,
        j3_num / j3_den,
    };
}

std::pair<std::vector<InvariantResult>, std::vector<bool>>
evaluate_batch_invariants(
    const Eigen::MatrixXd& points,
    const std::vector<std::array<int, 6>>& all_tuples,
    double denom_tol)
{
    std::vector<InvariantResult> invariants(all_tuples.size(), {0, 0, 0});
    std::vector<bool> valid(all_tuples.size(), false);

    for (size_t k = 0; k < all_tuples.size(); ++k) {
        auto result = evaluate_six_point_invariants(points, all_tuples[k], denom_tol);
        if (result.has_value()) {
            invariants[k] = result.value();
            valid[k] = true;
        }
    }
    return {invariants, valid};
}

}  // namespace capri
