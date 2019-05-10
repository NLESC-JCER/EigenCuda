# headers
include_directories(${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

add_library(eigencuda eigencuda.cc)
add_executable(benchmark benchmark.cc)

set_target_properties(eigencuda benchmark
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  )

target_compile_options(benchmark
  PRIVATE
  "-Wall"
)

target_link_libraries(benchmark
  PRIVATE
    eigencuda
    Eigen3::Eigen
    cxxopts::cxxopts
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
  )

if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()
