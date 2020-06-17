#ifndef CUDATENSOR_H_
#define CUDATENSOR_H_

#include <cuda_runtime.h>
#include <cudatensorbase.hpp>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>

/**
 * \brief Manage the tensor memory in the GPU
 */

namespace eigencuda {

class CudaTensor : CudaTensorBase {
 public:
  CudaTensor() = default;

  Index size() const override;

 private:
  // vector of dimensions;
  const Eigen::Tensor<float, 3>::Dimensions _dimensions;
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