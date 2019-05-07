#include "eigencuda.hpp"

namespace eigencuda {

  template <typename T>::EigenCuda() {cublasCreate(&_handle);}
  template <typename T>::EigenCuda(bool pinned) : _pinned{pinned} {
    cublasCreate(&_handle); }


template <typename T> EigenCuda<T>::~EigenCuda() {
  cublasDestroy(_handle);
  std::for_each(begin(_allocated), end(_allocated),
                [this](T *x) { this->fun_free(x); });
}

template <typename T>
void EigenCuda<T>::initialize_Matrices(Mat<T> &A, Mat<T> &B) {
  // Copy two matrices to the device

  // size of the Matrices
  std::size_t size_A = A.rows() * A.cols() * sizeof(T);
  std::size_t size_B = B.rows() * B.cols() * sizeof(T);

  // Pointers at the host
  T *hA = A.data();
  T *hB = B.data();

  // alloc memory on the GPU
  T *dA, *dB;

  // Allocate either pageable or pinned memory
  fun_alloc(&dA, size_A);
  fun_alloc(&dB, size_B);

  // Track the allocated variables
  _allocated.emplace_back(dA);
  _allocated.emplace_back(dB);

  // Transfer data to the GPU
  cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);
}

template <typename T>
void EigenCuda<T>::gemm(Mat<T> &A, Mat<T> &B, Mat<T> &C, cublasOperation_t op1 = CUBLAS_OP_N,
          cublasOperation_t op2 = CUBLAS_OP_N) {
  // call gemm from cublas
  if constexpr (std::is_same<float, T>()) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                pa, dA, A.rows(), dB, B.rows(), pb, dC, C.rows());
  } else if (std::is_same<double, T>()) {
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                pa, dA, A.rows(), dB, B.rows(), pb, dC, C.rows());
  }
}

template <typename T>
Mat<T> EigenCuda<T>::dot(Mat<T> A, Mat<T> B) {
  // Matrix multiplication

  // size of the resulting matrix
  std::size_t size_C = A.rows() * B.cols() * sizeof(T);
  Mat<T> C = Mat<T>::Zero(A.rows(), B.cols());
  T *hC = C.data();

  // allocate space in the device
  fun_alloc(&dC, size_C);
  _allocated.emplace_back(dC);

  // process on GPU
  gemm(A, B, C);

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, A.rows(), B.cols());

  return C;
}

// explicit instantiations
template class EigenCuda<float>;
template class EigenCuda<double>;
} // namespace eigencuda
