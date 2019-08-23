
add_library(eigencuda eigencuda.cc tensor_matrix.cc) 
add_executable(benchmark benchmark.cc)

target_include_directories(eigencuda
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
)

set_target_properties(benchmark eigencuda
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  )

target_compile_options(eigencuda
  PUBLIC
  "-Wall"
)

target_link_libraries(eigencuda
  PRIVATE
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
  )

target_link_libraries(benchmark
  PRIVATE
    Eigen3::Eigen
    eigencuda
  )

if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()
