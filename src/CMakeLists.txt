
add_library(eigencuda cudamatrix.cc cudapipeline.cc)

target_include_directories(eigencuda
  PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${CUDA_INCLUDE_DIRS}
)

set_target_properties(eigencuda
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  )

target_compile_options(eigencuda
  PUBLIC
  "-Wall"
)

target_link_libraries(eigencuda
  PUBLIC
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
  )

if(ENABLE_TESTING)
  add_subdirectory(tests)
endif()
