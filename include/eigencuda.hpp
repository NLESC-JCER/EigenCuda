#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <vector>

/**
 * \brief Perform matrix-matrix multiplication in a GPU
 *
 * The `EigenCuda` class handles the allocation and deallocation of arrays on
 * the GPU. Firstly, to perform a matrix multiplication, memory must be
 * allocated in the device to contain the involved matrices. The
 * `initialize_matrix_mem` method firstly allocates memory by calling the
 * `gpu_alloc` method that allocates either pinned or pageable memory, see:
 * https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/ Then the
 * array could be optionally copy to the device.
 */

namespace eigencuda {

inline cudaError_t checkCuda(cudaError_t result) {
// Check Cuda error
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
  }
#endif
  return result;
}

// Strides to batch gemm
struct Strides {
  long long int stA;
  long long int stB;
  long long int stC;
};

// Structure with the sizes to call ?GEMM
struct Shapes {
  int A_rows;
  int A_cols;
  int B_rows;
  int B_cols;
  int C_rows;

  Shapes(long int _a_rows, long int _a_cols, long int _b_rows, long int _b_cols,
         long int _c_rows)
      : A_rows{static_cast<int>(_a_rows)},
        A_cols{static_cast<int>(_a_cols)},
        B_rows{static_cast<int>(_b_rows)},
        B_cols{static_cast<int>(_b_cols)},
        C_rows{static_cast<int>(_c_rows)} {}
};

// col Major for CUDA
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
class EigenCuda {

 public:
  EigenCuda() {
    cublasCreate(&_handle);
    cudaStreamCreate(&_stream);
  }
  EigenCuda(bool pinned) : _pinned{pinned} {
    cublasCreate(&_handle);
    cudaStreamCreate(&_stream);
  }

  // Deallocate both the handler and allocated arrays
  ~EigenCuda();

  // Remove the copy operations
  EigenCuda(const EigenCuda &) = delete;
  EigenCuda &operator=(const EigenCuda &) = delete;

  // Matrix matrix multiplication
  Mat<T> dot(const Mat<T> &A, const Mat<T> &B) const;

  // Perform the triple matrix multiplication A * matrix * C, for the vector
  // of matrices given by tensor
  std::vector<Mat<T>> triple_tensor_product(const Mat<T> &A, const Mat<T> &C,
                                            const std::vector<Mat<T>> &tensor);

  // Perform a multiplication between a matrix and a tensor
  std::vector<Mat<T>> right_matrix_tensor(
      const Mat<T> &A, const std::vector<Mat<T>> &tensor) const;

 private:
  // Allocate memory in the device
  void gpu_alloc(T **x, std::size_t n) const;

  // Allocate memory for a tensor in the device;
  void gpu_alloc_tensor(T *arr[], int shape, int batchCount) const;

  // Deallocate memory from the device
  void gpu_free(T *x) const;

  // Free the memory allocated for a tensor
  void free_tensor_memory(T *arr[], int batchCount) const;

  // Copy a tensor to preallocated memory in the device
  void copy_tensor_to_dev(const std::vector<Mat<T>> &tensor, T *arr[]) const;

  // Allocate memory in the device, optionally copying the array to the GPU
  T *initialize_matrix_mem(const Mat<T> &A, bool copy_to_device = true) const;

  // Invoke the ?gemm function of cublas
  void gemm(Shapes shapes, const T *dA, const T *dB, T *dC) const;

  // Invoke the ?gemmBatched function of CuBlas.
  void gemmBatched(Shapes sh, T **dA, T **dB, T **dC, int batchCount) const;

  // Cuda variables
  cublasHandle_t _handle;
  bool _pinned = false;

  // Asynchronous stream
  cudaStream_t _stream;

  // Scalar constanst for calling blas
  T _alpha = 1.;
  T _beta = 0.;
  const T *_palpha = &_alpha;
  const T *_pbeta = &_beta;
};

}  // namespace eigencuda

#endif  // EIGENCUDA_H_
