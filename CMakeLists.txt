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

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Search for Cuda
find_package(CUDA REQUIRED)

# Search CXX options
find_package(cxxopts REQUIRED
  HINTS ${CONDA_PREFIX}/include
)

# search for Eigen
find_package (Eigen3 3.3 REQUIRED CONFIG
  HINTS ${CONDA_PREFIX}/include
)

message(STATUS "EIGEN Include: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
