cmake_minimum_required(VERSION 3.15)
project(pm_sampling CUDA CXX)

# Find required packages
find_package(CUDA REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Set pybind11 include path from virtualenv
set(PYBIND11_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../.venv/lib64/python3.11/site-packages/pybind11/include")
if(NOT EXISTS ${PYBIND11_INCLUDE_DIR})
    message(FATAL_ERROR "pybind11 headers not found at ${PYBIND11_INCLUDE_DIR}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/lib64
)

# Add library
add_library(pm_sampling MODULE
    pm_sampling_binding.cpp
    pm_sampling.cu
)

# Set Python extension suffix
set_target_properties(pm_sampling PROPERTIES PREFIX "")
if(APPLE)
    set_target_properties(pm_sampling PROPERTIES SUFFIX ".so")
endif()

# Link libraries
target_link_libraries(pm_sampling PRIVATE
    cupti
    cuda
    ${CUDA_LIBRARIES}
    ${Python_LIBRARIES}
)