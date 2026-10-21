cmake_minimum_required(VERSION 3.18)
project(capri_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Dependencies ──
find_package(Eigen3 3.3 REQUIRED)

# Optional: pybind11 for Python bindings
find_package(pybind11 REQUIRED)

# ── Core library ──
add_library(capri_core STATIC
    src/brackets.cpp
    src/six_point.cpp
    src/conditioning.cpp
    src/greedy.cpp
    src/stratification.cpp
    src/grassmann_plucker.cpp
    src/residual.cpp
)
target_include_directories(capri_core PUBLIC include)
target_link_libraries(capri_core PUBLIC Eigen3::Eigen)

# ── Python bindings (uncomment when ready) ──
pybind11_add_module(capri_cpp src/bindings.cpp)
target_link_libraries(capri_cpp PRIVATE capri_core)
