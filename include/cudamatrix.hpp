#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include "memory_manager.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
/*
 * \brief Perform Tensor-matrix multiplications in a GPU
 *
 * The `CudaPipeline` class handles the allocation and deallocation of arrays on
 * the GPU.
 */

namespace eigencuda {

class CudaMatrix {
 public:
  Index size() const { return _rows * _cols; };
  Index rows() const { return _rows; };
  Index cols() const { return _cols; };
  double *data() const { return _data.get(); };

  CudaMatrix(const Eigen::MatrixXd &matrix, const cudaStream_t &stream);

  // Allocate memory in the GPU for a matrix
  CudaMatrix(Index nrows, Index ncols, const cudaStream_t &stream);

  // Convert A Cudamatrix to an EigenMatrix
  operator Eigen::MatrixXd() const;

  void copy_to_gpu(const Eigen::MatrixXd &A);

 private:
  // Unique pointer with custom delete function
  using Unique_ptr_to_GPU_data = std::unique_ptr<double, void (*)(double *)>;

  Unique_ptr_to_GPU_data alloc_matrix_in_gpu(size_t size_arr) const;

  void throw_if_not_enough_memory_in_gpu(size_t requested_memory) const;

  size_t size_matrix() const { return this->size() * sizeof(double); }

  // Attributes of the matrix in the device
  Unique_ptr_to_GPU_data _data{nullptr,
                               [](double *x) { eigencuda::checkCuda(cudaFree(x)); }};
  cudaStream_t _stream = nullptr;
  Index _rows;
  Index _cols;
};

}  // namespace eigencuda

#endif  // EIGENCUDA_H_
