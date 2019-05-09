#include "eigencuda.hpp"

namespace eigencuda {

template <typename T> EigenCuda<T>::~EigenCuda() {
  cublasDestroy(_handle);
  for (auto &p: _allocated)
    this -> fun_free(p.second);
}

template <typename T>
void EigenCuda<T>::fun_alloc(T **x, std::size_t n) const {
  // Allocate memory in the device
  (_pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
}

template <typename T>
void EigenCuda<T>::fun_free(T *x) const {
  // Deallocate memory from the device
  (_pinned) ? cudaFreeHost(x) : cudaFree(x);
};

template <typename T>
unsigned EigenCuda<T>::initialize_Matrix(Mat<T> &A, bool copy_to_device) {
  // Copy two matrices to the device

  // size of the Matrices
  std::size_t size_A = A.rows() * A.cols() * sizeof(T);

  // Pointers at the host
  T *hA = A.data();

  // alloc memory on the GPU
  T *dA;

  // Allocate either pageable or pinned memory
  fun_alloc(&dA, size_A);

  // Track the allocated variables
  unsigned id = _counter;
  _allocated.emplace(std::make_pair(id, dA));
  _counter += 1;

  // Transfer data to the GPU
  if (copy_to_device)
    cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);

  return id;
}

template <typename T>
Mat<T> EigenCuda<T>::gemm(std::tuple<Mat<T>&, Mat<T>&, Mat<T>&> matrices,
			  std::tuple<unsigned, unsigned, unsigned> ids) {
  // Invoke the gemm subroutine from cublas
  Mat<T> A, B, C;
  unsigned id_A, id_B, id_C;
  std::tie(A, B, C) = matrices;
  std::tie(id_A, id_B, id_C) = ids;

  T *dA, *dB, *dC;
  dA = _allocated[id_A];
  dB = _allocated[id_B];
  dC = _allocated[id_C];

  // call gemm from cublas
  if constexpr (std::is_same<float, T>()) {
      cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
		  _pa, dA, A.rows(), dB, B.rows(), _pb, dC, C.rows());
    } else if (std::is_same<double, T>()) {
    cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                _pa, dA, A.rows(), dB, B.rows(), _pb, dC, C.rows());
  }
  return C;
}

template <typename T> Mat<T> EigenCuda<T>::dot(Mat<T> &A, Mat<T> &B) {
  // Matrix multiplication
 
  // Matrix to store the result
  Mat<T> C = Mat<T>::Zero(A.rows(), B.cols());
  std::size_t size_C = C.rows() * C.cols() * sizeof(T);

  // Id of the Arrays to compute the multiplication
  std::tuple<unsigned, unsigned, unsigned> ids = 
    std::make_tuple(initialize_Matrix(A),
		    initialize_Matrix(B),
		    initialize_Matrix(C, false)
		    );

  // process on GPU
  std::tuple< Mat<T>&, Mat<T>&, Mat<T>& > matrices = std::forward_as_tuple(A, B, C);
  gemm(matrices, ids);

  // send data back to CPU
  T *hC = C.data();
  T *dC = _allocated[std::get<2>(ids)];
  cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, A.rows(), B.cols());

  return C;
}

// explicit instantiations
template class EigenCuda<float>;
template class EigenCuda<double>;
} // namespace eigencuda
