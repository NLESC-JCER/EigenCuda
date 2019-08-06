#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cublas_v2.h>
#include <curand.h>
#include <tuple>
#include <unordered_map>
#include <vector>

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

// Structure with the sizes to call ?GEMM
struct Shapes {
  int A_rows;
  int A_cols;
  int B_rows;
  int B_cols;
  int C_rows;
};

// col Major for CUDA
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T> class EigenCuda {

public:
  EigenCuda() { cublasCreate(&_handle); }
  EigenCuda(bool pinned) : _pinned{pinned} { cublasCreate(&_handle); }

  // Deallocate both the handler and allocated arrays
  ~EigenCuda();

  // Remove the copy operations
  EigenCuda(const EigenCuda &) = delete;
  EigenCuda &operator=(const EigenCuda &) = delete;

  // Matrix matrix multiplication
  Mat<T> dot(Mat<T> &A, Mat<T> &B);

  // Perform the triple matrix multiplication A * matrix * C, for the vector
  // of matrices given by tensor
  std::vector<Mat<T>> triple_tensor_product(Mat<T> &A, Mat<T> &C,
                                            std::vector<Mat<T>> &tensor);

private:
  // Allocate memory in the device
  void fun_alloc(T **x, std::size_t n) const;

  // Deallocate memory from the device
  void fun_free(T *x) const;

  // Copy matricex to the device
  unsigned initialize_Matrix(Mat<T> &A, bool copy_to_device = true);

  // Invoke the ?gemm function of cublas
  void gemm(Shapes shapes, std::tuple<unsigned, unsigned, unsigned> ids);

  // Deallocate certain matrix from the device
  void free_matrix(unsigned id);

  // Cuda variables
  cublasHandle_t _handle;
  bool _pinned = false;

  // Allocation booking
  unsigned _counter = 0;
  std::unordered_map<unsigned, T *> _allocated;

  // Scalar constanst for calling blas
  T _alpha = 1.;
  T _beta = 0.;
  const T *_pa = &_alpha;
  const T *_pb = &_beta;
};

} // namespace eigencuda

#endif // EIGENCUDA_H_
