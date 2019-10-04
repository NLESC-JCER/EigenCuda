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
 * The `CudaPipeline` class handles the allocation and deallocation of arrays on
 * the GPU.
 */

namespace eigencuda {
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
  double *pointer() const { return _pointer.get(); };

  CudaMatrix(const Eigen::MatrixXd &matrix, const cudaStream_t &stream)
      : _rows{static_cast<int>(matrix.rows())},
        _cols{static_cast<int>(matrix.cols())} {
    size_t size_matrix = this->size() * sizeof(double);
    _pointer = std::move(alloc_matrix_in_gpu(size_matrix));
    cudaError_t err =
        cudaMemcpyAsync(_pointer.get(), matrix.data(), size_matrix,
                        cudaMemcpyHostToDevice, stream);
    if (err != 0) {
      throw std::runtime_error("Error copy arrays to device");
    }
  }

  // Unique pointer with custom delete function
  using double_unique_ptr = std::unique_ptr<double, void (*)(double *)>;

 private:
  friend class CudaPipeline;

  CudaMatrix(long int nrows, long int ncols)
      : _rows{static_cast<int>(nrows)}, _cols{static_cast<int>(ncols)} {
    size_t size_matrix = this->size() * sizeof(double);
    _pointer = std::move(alloc_matrix_in_gpu(size_matrix));
  }

  double_unique_ptr alloc_matrix_in_gpu(size_t size_matrix) const;

  double_unique_ptr _pointer{nullptr,
                             [](double *x) { checkCuda(cudaFree(x)); }};
  int _rows;
  int _cols;
};

void free_mem_in_gpu(double *x);

/* \brief The CudaPipeline class offload Eigen operations to an *Nvidia* GPU
 * using the CUDA language. The Cublas handle is the context manager for all the
 * resources needed by Cublas. While a stream is a queue of sequential
 * operations executed in the Nvidia device.
 */
class CudaPipeline {
 public:
  CudaPipeline() {
    cublasCreate(&_handle);
    cudaStreamCreate(&_stream);
  }
  ~CudaPipeline();

  CudaPipeline(const CudaPipeline &) = delete;
  CudaPipeline &operator=(const CudaPipeline &) = delete;

  // Perform a multiplication between a matrix and a tensor
  void right_matrix_tensor_mult(std::vector<Eigen::MatrixXd> &tensor,
                                const Eigen::MatrixXd &A) const;

  // Perform matrix1 * matrix2
  Eigen::MatrixXd matrix_mult(const Eigen::MatrixXd &A,
                              const Eigen::MatrixXd &B) const;

  const cudaStream_t &get_stream() const { return _stream; };

 private:
  void throw_if_not_enough_memory_in_gpu(size_t required) const;

  // Invoke the ?gemm function of cublas
  void gemm(const CudaMatrix &A, const CudaMatrix &B, CudaMatrix &C) const;

  // The cublas handles allocates hardware resources on the host and device.
  cublasHandle_t _handle;

  // Asynchronous stream
  cudaStream_t _stream;
};

}  // namespace eigencuda

#endif  // EIGENCUDA_H_
