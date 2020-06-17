#ifndef CUDATENSORBASE_H_
#define CUDATENSORBASE_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <memory>
/**
 * \brief Manage the tensor memory on the GPU
 */

namespace eigencuda {

// Check the return value of the cuda operations
cudaError_t checkCuda(cudaError_t result);

// Int64
using Index = Eigen::Index;

/**
 * \brief Number of GPUs on the host
 */
Index count_available_gpus();

class CudaTensorBase {
 public:
  // Pointer to the GPU memory
  double *data() const { return _data.get(); };

  /**
   * \brief calculate the total number of elements
   */
  Index virtual size() const = 0;

 protected:
  // Unique pointer with custom delete function
  using Unique_ptr_to_GPU_data = std::unique_ptr<double, void (*)(double *)>;

  // GPU stream
  cudaStream_t _stream = nullptr;

  // Attributes of the matrix in the device
  Unique_ptr_to_GPU_data _data{
      nullptr, [](double *x) { eigencuda::checkCuda(cudaFree(x)); }};

  size_t size_tensor() const { return this->size() * sizeof(double); }

  /**
   * \brief reserve space for the tensor in the GPU
   * @param size of the tensor
   */
  Unique_ptr_to_GPU_data alloc_tensor_in_gpu(size_t) const;

  /**
   * \brief Check that the GPU has enough memory
   * @param requested_memory
   */
  void throw_if_not_enough_memory_in_gpu(size_t requested_memory) const;
};

}  // namespace eigencuda

#endif  // CUDATENSORBASE_H_