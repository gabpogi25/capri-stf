// ─────────────────────────────────────────────────────────────
// capri_cpp / bindings.cpp
//
// pybind11 bindings to expose C++ CAPRI core to Python.
//
// Uncomment and build when ready:
//   cd cpp_core && mkdir build && cd build
//   cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
//   cmake --build . --config Release
//
// Then from Python:
//   import capri_cpp
//   score = capri_cpp.compute_conditioning_score(points, tuple)
// ─────────────────────────────────────────────────────────────

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "capri/brackets.h"
#include "capri/six_point.h"
#include "capri/conditioning.h"
#include "capri/greedy.h"

#include "capri/stratification.h"
#include "capri/grassmann_plucker.h"
#include "capri/residual.h"
#include "capri/pencil.h"

namespace py = pybind11;

PYBIND11_MODULE(capri_cpp, m) {
    m.doc() = "CAPRI C++ core — high-performance projective invariant computation";

    // --- Stratification Enums ---
    py::enum_<capri::DegeneracyStratum>(m, "DegeneracyStratum")
        .value("GENERIC", capri::DegeneracyStratum::GENERIC)
        .value("BOUNDARY", capri::DegeneracyStratum::BOUNDARY)
        .value("COMPOUND", capri::DegeneracyStratum::COMPOUND)
        .value("CRITICAL", capri::DegeneracyStratum::CRITICAL)
        .export_values();

    py::class_<capri::InvariantResult>(m, "InvariantResult")
        .def_readwrite("j1", &capri::InvariantResult::j1)
        .def_readwrite("j2", &capri::InvariantResult::j2)
        .def_readwrite("j3", &capri::InvariantResult::j3);

    m.def("compute_bracket", &capri::compute_bracket,
          "Compute a single bracket [pi pj pk pl]",
          py::arg("points"), py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"));

    m.def("batch_invariant_noise_gain", &capri::batch_invariant_noise_gain,
          "Exact kappa_J for (T, 6) tuples of (4, n) points (unit representatives)",
          py::arg("points"), py::arg("tuples"));
    m.def("log_abs_brackets", &capri::log_abs_brackets,
          "log|det| of unit-normalized (Q, 4) column quadruples",
          py::arg("points"), py::arg("quads"));

    m.def("compute_all_brackets", &capri::compute_all_brackets,
          "Compute all 15 brackets for a 6-point tuple");

    m.def("compute_conditioning_score", &capri::compute_conditioning_score,
          "Compute L(S) conditioning score for a tuple");

    m.def("compute_conditioning_scores_batched", &capri::compute_conditioning_scores_batched,
          "Batched conditioning scores for K tuples");

    m.def("compute_chart_free_conditioning", &capri::compute_chart_free_conditioning,
          "Compute chart-free conditioning score \kappa_J(x)",
          py::arg("points"), py::arg("tuple_indices"), py::arg("noise_sigma") = 1e-4);

    m.def("evaluate_six_point_invariants", &capri::evaluate_six_point_invariants,
          "Compute (J1, J2, J3) for a 6-point tuple",
          py::arg("points"), py::arg("tuple_indices"), py::arg("denom_tol") = 1e-10);

    m.def("select_tuples_greedy", &capri::select_tuples_greedy,
          "Lazy greedy submodular tuple selection",
          py::arg("points"),
          py::arg("max_tuples") = 500,
          py::arg("conditioning_threshold") = 1000.0,
          py::arg("coverage_weight") = 1.0,
          py::arg("penalty_weight") = 0.001,
          py::arg("candidate_sample_size") = 10000,
          py::arg("seed") = 0);

    // --- Stratification ---
    m.def("classify_degeneracy_stratum", &capri::classify_degeneracy_stratum,
          "Classify the degeneracy stratum of a 6-point tuple",
          py::arg("points"), py::arg("tuple_indices"),
          py::arg("near_zero_threshold") = 1e-6, py::arg("boundary_threshold") = 1e-2);

    m.def("batch_classify_strata", &capri::batch_classify_strata,
          "Batch classify strata for K tuples",
          py::arg("points"), py::arg("all_tuples"),
          py::arg("near_zero_threshold") = 1e-6, py::arg("boundary_threshold") = 1e-2);

    // --- Syzygy (Grassmann-Plucker) ---
    m.def("compute_plucker_residual", &capri::compute_plucker_residual,
          "Compute maximum absolute Plucker residual for a tuple",
          py::arg("points"), py::arg("tuple_indices"));

    m.def("batch_plucker_residuals", &capri::batch_plucker_residuals,
          "Batch compute Plucker residuals for K tuples",
          py::arg("points"), py::arg("all_tuples"));

    // --- Syzygy (Residual) ---
    m.def("compute_syzygy_residual", &capri::compute_syzygy_residual,
          "Compute the Plucker syzygy consistency residual (rho) from 15 brackets",
          py::arg("brackets"));

    m.def("compute_exact_admissibility_bound", &capri::compute_exact_admissibility_bound,
          "Compute the exact admissibility bound tau_exact(delta) = 3*eps*(2+eps)",
          py::arg("delta_drift"));
}
