#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

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
 * The `EigenCuda` class handles the allocation and deallocation of arrays on
 * the GPU.
 */

namespace eigencuda {
// Unique pointer with custom delete function
using uniq_double = std::unique_ptr<double, void (*)(double *)>;

inline cudaError_t checkCuda(cudaError_t result) {
// Check Cuda error
#if defined(DEBUG)
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
  }
#endif
  return result;
}

// Data of the matrix stored in the GPU
class CudaMatrix {
 public:
  int size() const { return _rows * _cols; };
  int rows() const { return _rows; };
  int cols() const { return _cols; };
  double *ptr() const { return _ptr.get(); };

  CudaMatrix(uniq_double &&ptr, long int nrows, long int ncols)
      : _ptr{std::move(ptr)},
        _rows{static_cast<int>(nrows)},
        _cols{static_cast<int>(ncols)} {}

 private:
  uniq_double _ptr;
  int _rows;
  int _cols;
};

// Delete allocated memory in the GPU
void free_mem_in_gpu(double *x);

class EigenCuda {
 public:
  EigenCuda() {
    cublasCreate(&_handle);
    cudaStreamCreate(&_stream);
  }
  ~EigenCuda();

  EigenCuda(const EigenCuda &) = delete;
  EigenCuda &operator=(const EigenCuda &) = delete;

  // Allocate memory for a matrix and copy it to the device
  uniq_double copy_matrix_to_gpu(const Eigen::MatrixXd &matrix) const;

  // Matrix matrix multiplication
  Eigen::MatrixXd matrix_mult(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) const;

  // Perform a multiplication between a matrix and a tensor
  void right_matrix_tensor_mult(std::vector<Eigen::MatrixXd> &&tensor,
                                const Eigen::MatrixXd &A) const;

 private:
  void check_available_memory_in_gpu(size_t required) const;
  uniq_double alloc_matrix_in_gpu(size_t size_matrix) const;

  // Invoke the ?gemm function of cublas
  void gemm(const CudaMatrix &A, const CudaMatrix &B, CudaMatrix &C) const;

  // The cublas handles allocates hardware resources on the host and device.
  cublasHandle_t _handle;

  // Asynchronous stream
  cudaStream_t _stream;
};

}  // namespace eigencuda

#endif  // EIGENCUDA_H_
