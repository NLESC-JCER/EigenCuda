# headers
include_directories(${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

add_executable(eigen_cuda gemm_cublas.cc)

set_target_properties(eigen_cuda
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  )

target_compile_options(eigen_cuda
  PRIVATE
  "-Wall"
)

target_link_libraries(eigen_cuda
  PRIVATE
    Eigen3::Eigen
    cxxopts::cxxopts
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
  )
