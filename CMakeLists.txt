# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.5)

# Project's name
project(EigenCuda LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

# User input options
option(ENABLE_TESTING "Build and enable tests" OFF)
if(ENABLE_TESTING)
  enable_testing()
  find_package(Boost REQUIRED COMPONENTS unit_test_framework)
endif(ENABLE_TESTING)

# Search for Cuda
find_package(CUDA REQUIRED)

# search for Eigen
find_package (Eigen3 3.3 REQUIRED CONFIG
  HINTS ${CONDA_PREFIX}/include
)

message(STATUS "EIGEN Include: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
