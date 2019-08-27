#include "eigencuda.hpp"

namespace eigencuda {

template <typename T>
TensorMatrix<T>::~TensorMatrix() {
  this->free_tensor_memory(_tensorA, _batchCount);
  this->free_tensor_memory(_tensorB, _batchCount);
  this->free_tensor_memory(_tensorC, _batchCount);
}

template <typename T>
std::vector<Mat<T>> TensorMatrix<T>::tensor_dot_matrix(
    std::vector<Mat<T>> tensor, Mat<T> B) {

  // First submatrix from the tensor
  Mat<T> matrix = tensor[0];

  // copy to the device the input tensor
  this->copy_tensor_to_dev(tensor, _tensorA);

  // represent the matrix B as a tensor where all the submatrices are the same
  T *mtxB;
  size_t size_B = B.size() * sizeof(T);
  this->gpu_alloc(&mtxB, size_B);
  cudaMemcpyAsync(mtxB, B.data(), size_B, cudaMemcpyHostToDevice,
                  this->_stream);
  for (auto i = 0; i < _batchCount; i++) {
    _tensorB[i] = mtxB;
  }

  // Allocate space in the device for the array of pointers
  const T **dA, **dB;
  T **dC;
  size_t size_batch = _batchCount * sizeof(T *);
  cudaMalloc(&dA, size_batch);
  cudaMalloc(&dB, size_batch);
  cudaMalloc(&dC, size_batch);

  // Copy the arrays of pointers from host to the device
  cudaMemcpyAsync(dA, _tensorA, size_batch, cudaMemcpyHostToDevice,
                  this->_stream);
  cudaMemcpyAsync(dB, _tensorB, size_batch, cudaMemcpyHostToDevice,
                  this->_stream);
  cudaMemcpyAsync(dC, _tensorC, size_batch, cudaMemcpyHostToDevice,
                  this->_stream);

  // Call tensor matrix multiplication
  Shapes sh{matrix.rows(), matrix.cols(), B.rows(), B.cols(), matrix.rows()};
  this->gemmBatched(sh, dA, dB, dC, _batchCount);

  // Vector containing the results
  std::vector<Mat<T>> rs(_batchCount, Mat<T>::Zero(matrix.rows(), B.cols()));
  std::size_t size_out = matrix.rows() * B.cols() * sizeof(T);

  // Copy Array of pointers on the device to the host
  cudaMemcpyAsync(_tensorC, dC, size_batch, cudaMemcpyDeviceToHost,
                  this->_stream);

  // Copy each array back to the device
  for (auto i = 0; i < _batchCount; i++) {
    T *hout = rs[i].data();
    T *dout = _tensorC[i];
    cudaMemcpyAsync(hout, dout, size_out, cudaMemcpyDeviceToHost,
                    this->_stream);
    rs[i] = Eigen::Map<Mat<T>>(hout, matrix.rows(), B.cols());
    ;
  }

  // Deallocate all the memory from the device
  this->gpu_free(mtxB);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  return rs;
}

// explicit instantiations
template class TensorMatrix<float>;
template class TensorMatrix<double>;
}  // namespace eigencuda
