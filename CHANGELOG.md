# Change log

# [0.2.0] 27/08/2019
### Added
 - Tensor matrix multiplacation using [gemmbatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched).
 - [Async calls](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79) to memory copies.
 - Properly free memory after the tensor operation is done.

# [0.1.0]

### New
 - Use a template function to perform matrix matrix multiplacation using [cublas](https://docs.nvidia.com/cuda/cublas/index.html).
 - Use either *pinned* (**default**) or *pageable* memory, see [cuda optimizations](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/).
