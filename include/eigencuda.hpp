#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <curand.h>
#include <deque>

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

// col Major for CUDA
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T> class EigenCuda {

public:
  EigenCuda();
  EigenCuda(bool pinned);

  // Deallocate both the handler and allocated arrays
  ~EigenCuda();

  // Remove the copy operations
  EigenCuda(const EigenCuda &) = delete;
  EigenCuda &operator=(const EigenCuda &) = delete;


  void fun_alloc(T **x, std::size_t n) {
    // Allocate memory in the device
    (_pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
  }

  void fun_free(T *x) {
    // Deallocate memory from the device
    (_pinned) ? cudaFreeHost(x) : cudaFree(x);
  };

  // Copy two matrices to the device
  void initialize_Matrices(Mat<T> A, Mat<T> B);

  // Invoke the ?gemm function of cublas
  Mat<T> gemm(Mat<T> A, Mat<T> B, Mat<T> C);

  // Matrix multiplication
  Mat<T> dot(Mat<T> A, Mat<T> B);

private:
  cublasHandle_t _handle;
  bool _pinned = false;
  std::deque<T *> _allocated;

  // Scalar constanst for calling blas
  T _alpha = 1.;
  T _beta = 0.;
  const T *_pa = &_alpha;
  const T *_pb = &_beta;
};

template <typename T>
Mat<T> cublas_gemm(Mat<T> &A, Mat<T> &B, bool pinned = false) {
  // Transfer the matrix matrix multiplacation of Eigen to GPU, using
  // CUBLas

  // Scalar constanst for calling blas
  constexpr T alpha = 1.;
  constexpr T beta = 0.;
  const T *pa = &alpha;
  const T *pb = &beta;

  // Size of the Matrices
  std::size_t size_A = A.rows() * A.cols() * sizeof(T);
  std::size_t size_B = A.cols() * B.cols() * sizeof(T);
  std::size_t size_C = A.rows() * B.cols() * sizeof(T);

  Mat<T> C = Mat<T>::Zero(A.rows(), B.cols());

  // and their pointers
  T *hA = A.data();
  T *hB = B.data();
  T *hC = C.data();

  // alloc memory on the GPU
  T *dA, *dB, *dC;

  // Allocate either pageable or pinned memory
  auto fun_alloc = [&pinned](T **x, std::size_t n) {
    (pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
  };

  fun_alloc(&dA, size_A);
  fun_alloc(&dB, size_B);
  fun_alloc(&dC, size_C);

  // cuda handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Transfer data to GPU
  cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);

  // process on GPU
  if constexpr (std::is_same<float, T>()) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                pa, dA, A.rows(), dB, B.rows(), pb, dC, C.rows());
  } else if (std::is_same<double, T>()) {
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                pa, dA, A.rows(), dB, B.rows(), pb, dC, C.rows());
  }

  // send data back to CPU
  cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, A.rows(), B.cols());

  // free memory
  cublasDestroy(handle);

  auto fun_free = [&pinned](T *x) { (pinned) ? cudaFreeHost(x) : cudaFree(x); };

  fun_free(dA);
  fun_free(dB);
  fun_free(dC);

  return C;
}

template <typename T>
Mat<T> triple_product(Mat<T> &A, Mat<T> &B, Mat<T> &C, bool pinned = false) {
  // Perform the triple matrix Multiplication: A^T * B * C

  // Transfer the matrix matrix multiplacation of Eigen to GPU, using
  // CUBLas

  // Scalar constanst for calling blas
  constexpr T alpha = 1.;
  constexpr T beta = 0.;
  const T *pa = &alpha;
  const T *pb = &beta;

  // Size of the Matrices
  std::size_t size_A = A.rows() * A.cols() * sizeof(T);
  std::size_t size_B = B.rows() * B.cols() * sizeof(T);
  std::size_t size_C = C.rows() * C.cols() * sizeof(T);
  std::size_t size_X = B.rows() * C.cols() * sizeof(T);
  std::size_t size_Y = A.cols() * C.cols() * sizeof(T);

  Mat<T> X = Mat<T>::Zero(B.rows(), C.cols());
  Mat<T> Y = Mat<T>::Zero(A.cols(), C.cols());

  // and their pointers
  T *hA = A.data();
  T *hB = B.data();
  T *hC = C.data();
  T *hY = C.data();

  // alloc memory on the GPU
  T *dA, *dB, *dC, *dX, *dY;

  // Allocate either pageable or pinned memory
  auto fun_alloc = [&pinned](T **x, std::size_t n) {
    (pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
  };

  fun_alloc(&dA, size_A);
  fun_alloc(&dB, size_B);
  fun_alloc(&dC, size_C);
  fun_alloc(&dX, size_X);
  fun_alloc(&dY, size_Y);

  // cuda handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Transfer data to GPU
  cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC, size_C, cudaMemcpyHostToDevice);

  // multiplied in the GPU
  if constexpr (std::is_same<float, T>()) {
    // X = B * C
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.rows(), C.cols(), B.cols(),
                pa, dB, B.rows(), dC, C.rows(), pb, dX, X.rows());
    // R = A^T * X
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.cols(), X.cols(), A.rows(),
                pa, dA, A.rows(), dX, X.rows(), pb, dY, Y.rows());
  } else if (std::is_same<double, T>()) {
    // X = B * C
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.rows(), C.cols(), B.cols(),
                pa, dB, B.rows(), dC, C.rows(), pb, dX, X.rows());
    // R = A^T * X
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.cols(), X.cols(), A.rows(),
                pa, dA, A.rows(), dX, X.rows(), pb, dY, Y.rows());
  }
  // send data back to CPU
  cudaMemcpy(hY, dY, size_Y, cudaMemcpyDeviceToHost);

  // create an eigen matrix
  Y = Eigen::Map<Mat<T>>(hY, A.cols(), C.cols());

  // free memory
  cublasDestroy(handle);

  auto fun_free = [&pinned](T *x) { (pinned) ? cudaFreeHost(x) : cudaFree(x); };

  fun_free(dA);
  fun_free(dB);
  fun_free(dC);
  fun_free(dX);
  fun_free(dY);

  return Y;
}

} // namespace eigencuda

#endif // EIGENCUDA_H_
