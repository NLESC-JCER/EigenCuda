# Change log

# [0.4.0] 10/02/2020
### Changed
  - Split the memory management (`CudaMatrix`) from the CUBLAS invocation (`CudaPipeline`)
  - Moved all the allocation to the smart pointers inside `CudaMatrix`
 - Removed unused headers

# [0.3.0] 26/09/2019
### Added
 - Smart pointers to handle cuda resources
 - New CudaMatrix class
 - Use Eigen::MatrixXd
 - Check available memory in the GPU before computing

### Removed
 - Template class, implementation only for double available
 - Triple tensor product
 - Shapes struct


# [0.2.0] 27/08/2019
### Added
 - Tensor matrix multiplacation using [gemmbatched](https://docs.nvidia.com/cuda/CUBLAS/index.html#CUBLAS-lt-t-gt-gemmbatched).
 - [Async calls](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79) to memory copies.
 - Properly free memory after the tensor operation is done.

# [0.1.0]

### New
 - Use a template function to perform matrix matrix multiplacation using [CUBLAS](https://docs.nvidia.com/cuda/CUBLAS/index.html).
 - Use either *pinned* (**default**) or *pageable* memory, see [cuda optimizations](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/).
