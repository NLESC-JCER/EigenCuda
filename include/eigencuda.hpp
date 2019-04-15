#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstdlib>
#include <cublas_v2.h>
#include <curand.h>

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

template <typename T>
Mat<T> cublas_gemm(Mat<T> A, Mat<T> B, bool pinned = false) {
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

}

#endif  // EIGENCUDA_H_
