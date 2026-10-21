#pragma once
// Exact kappa_J and batched log-brackets for the pencil test
// (docs/MATHEMATICS.md, Sections 2-5).
#include <pybind11/numpy.h>

namespace capri {
namespace py = pybind11;

// kappa_J = max_k |grad log|J_k|| on unit representatives, for T tuples.
// points: (4, n) float64; tuples: (T, 6) int64. Returns (T,), inf for a singular bracket.
py::array_t<double> batch_invariant_noise_gain(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<long long, py::array::c_style | py::array::forcecast> tuples);

// log|det| of the unit-normalized columns of each quadruple. quads: (Q, 4) int64.
py::array_t<double> log_abs_brackets(
    py::array_t<double, py::array::c_style | py::array::forcecast> points,
    py::array_t<long long, py::array::c_style | py::array::forcecast> quads);
}  // namespace capri
