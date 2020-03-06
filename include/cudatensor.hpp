#ifndef CUDATENSOR_H_
#define CUDATENSOR_H_

#include <cuda_runtime.h>
#include <cudatensorbase.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

/**
 * \brief Manage the tensor memory in the GPU
 */

namespace eigencuda {

/* // Check the return value of the cuda operations
cudaError_t checkCuda(cudaError_t result);

// Unique pointer with custom delete function
using Unique_ptr_to_GPU_data = std::unique_ptr<double, void (*)(double *)>;

// Int64
using Index = Eigen::Index;
 */

// /**
//  * \brief Number of GPUs on the host
//  */
// Index count_available_gpus();

class CudaTensor : CudaTensorBase {
 public:
  CudaTensor() = default;

  Index size() const override { return 0; };

 protected:
  /**
   * \brief Copy the tensor to the GPU
   */
  void copy_to_gpu(const Eigen::Tensor<double, 3> &A);

  /**
   * \brief Convert A CudaTensor to an EigenTensor
   */
  operator Eigen::Tensor<double, 3>() const;
};

}  // namespace eigencuda

#endif  // CUDATENSOR_H_