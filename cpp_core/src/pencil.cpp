// capri_cpp / pencil.cpp -- see include/capri/pencil.h
#include "capri/pencil.h"
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace capri {

namespace {
// J1, J2, J3: (numerator quad 1, numerator quad 2, denominator quad 1, denominator quad 2),
// positions 0..5 within the tuple; same order as algorithms/conditioning/scores.py::_J_QUADS.
const int J_QUADS[3][4][4] = {
    {{0, 1, 2, 3}, {0, 1, 4, 5}, {0, 1, 2, 4}, {0, 1, 3, 5}},
    {{0, 2, 3, 4}, {1, 2, 3, 5}, {0, 2, 3, 5}, {1, 2, 3, 4}},
    {{0, 1, 4, 5}, {2, 3, 4, 5}, {0, 2, 4, 5}, {1, 3, 4, 5}},
};
const double SIGNS[4] = {1.0, 1.0, -1.0, -1.0};
// Same tolerance as algorithms/conditioning/scores.py::SINGULAR_BRACKET_TOL.
const double SINGULAR_BRACKET_TOL = 1e-12;

std::vector<Eigen::Vector4d> unit_columns(const py::array_t<double, py::array::c_style | py::array::forcecast>& points) {
    auto P = points.unchecked<2>();
    if (P.shape(0) != 4) throw std::invalid_argument("points must have shape (4, n)");
    std::vector<Eigen::Vector4d> cols(P.shape(1));
    for (py::ssize_t c = 0; c < P.shape(1); ++c) {
        Eigen::Vector4d v(P(0, c), P(1, c), P(2, c), P(3, c));
        double norm = v.norm();
        if (norm <= 0.0) throw std::invalid_argument("zero-norm homogeneous point");
        cols[c] = v / norm;
    }
    return cols;
}
}  // namespace

py::array_t<double> batch_invariant_noise_gain(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<long long, py::array::c_style | py::array::forcecast> tuples) {
    const auto cols = unit_columns(points);
    auto T = tuples.unchecked<2>();
    if (T.shape(1) != 6) throw std::invalid_argument("tuples must have shape (T, 6)");
    const py::ssize_t n_tuples = T.shape(0);
    py::array_t<double> out(n_tuples);
    auto gains = out.mutable_unchecked<1>();
    const double inf = std::numeric_limits<double>::infinity();
    {
        py::gil_scoped_release release;
        for (py::ssize_t t = 0; t < n_tuples; ++t) {
            double best = 0.0;
            bool singular = false;
            for (int k = 0; k < 3 && !singular; ++k) {
                Eigen::Matrix<double, 4, 6> g = Eigen::Matrix<double, 4, 6>::Zero();
                for (int b = 0; b < 4; ++b) {
                    Eigen::Matrix4d M;
                    for (int m = 0; m < 4; ++m) M.col(m) = cols[T(t, J_QUADS[k][b][m])];
                    const double det = M.determinant();
                    if (std::abs(det) < SINGULAR_BRACKET_TOL) { singular = true; break; }
                    const Eigen::Matrix4d inv = M.inverse();          // closed form for fixed 4x4
                    for (int m = 0; m < 4; ++m)                        // gradient wrt column m = row m of M^{-1}
                        g.col(J_QUADS[k][b][m]) += SIGNS[b] * inv.row(m).transpose();
                }
                if (!singular) best = std::max(best, g.norm());
            }
            gains(t) = singular ? inf : best;
        }
    }
    return out;
}

py::array_t<double> log_abs_brackets(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<long long, py::array::c_style | py::array::forcecast> quads) {
    const auto cols = unit_columns(points);
    auto Q = quads.unchecked<2>();
    if (Q.shape(1) != 4) throw std::invalid_argument("quads must have shape (Q, 4)");
    py::array_t<double> out(Q.shape(0));
    auto logs = out.mutable_unchecked<1>();
    {
        py::gil_scoped_release release;
        for (py::ssize_t q = 0; q < Q.shape(0); ++q) {
            Eigen::Matrix4d M;
            for (int m = 0; m < 4; ++m) M.col(m) = cols[Q(q, m)];
            logs(q) = std::log(std::abs(M.determinant()));
        }
    }
    return out;
}

}  // namespace capri
