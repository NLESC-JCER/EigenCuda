add_executable(gemm_cublas gemm_cublas.cpp)

set_target_properties(gemm_cublas
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  )

target_compile_options(gemm_cublas
  PRIVATE
  "-Wall"
)

target_link_libraries(gemm_cublas
  PRIVATE
    Eigen3::Eigen
    cxxopts::cxxopts
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
  )
