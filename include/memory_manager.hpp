#ifndef MEMORY_MANAGER_H_
#define MEMORY_MANAGER_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <memory>
/**
 * \brief Perform Tensor contractions using cutensor
 */

namespace eigencuda {

// Check the return value of the cuda operations
cudaError_t checkCuda(cudaError_t result);

// Unique pointer with custom delete function
using Unique_ptr_to_GPU_data = std::unique_ptr<double, void (*)(double *)>;

// Int64
using Index = Eigen::Index;

/**
 * \brief Number of GPUs on the host
 */
Index count_available_gpus();
/**
 * \brief reserve space for the tensor in the GPU
 * @param size of the tensor
 */
Unique_ptr_to_GPU_data alloc_tensor_in_gpu(size_t);

void throw_if_not_enough_memory_in_gpu(size_t requested_memory);

}  // namespace eigencuda

#endif  // MEMORY_MANAGER_H_