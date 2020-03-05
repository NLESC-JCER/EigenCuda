#include "cudamatrix.hpp"

namespace eigencuda {

CudaMatrix::CudaMatrix(const Eigen::MatrixXd &matrix,
                       const cudaStream_t &stream)
    : _rows{static_cast<Index>(matrix.rows())},
      _cols{static_cast<Index>(matrix.cols())} {
  _data = eigencuda::alloc_tensor_in_gpu(
      size_matrix());  // alloc_matrix_in_gpu(size_matrix());
  _stream = stream;
  cudaError_t err = cudaMemcpyAsync(_data.get(), matrix.data(), size_matrix(),
                                    cudaMemcpyHostToDevice, stream);
  if (err != 0) {
    throw std::runtime_error("Error copy arrays to device");
  }
}

CudaMatrix::CudaMatrix(Index nrows, Index ncols, const cudaStream_t &stream)
    : _rows{static_cast<Index>(nrows)}, _cols{static_cast<Index>(ncols)} {
  _data = alloc_matrix_in_gpu(size_matrix());
  _stream = stream;
}

CudaMatrix::operator Eigen::MatrixXd() const {
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(this->rows(), this->cols());
  eigencuda::checkCuda(cudaMemcpyAsync(result.data(), this->data(), this->size_matrix(),
                            cudaMemcpyDeviceToHost, this->_stream));
  eigencuda::checkCuda(cudaStreamSynchronize(this->_stream));
  return result;
}

void CudaMatrix::copy_to_gpu(const Eigen::MatrixXd &A) {
  size_t size_A = static_cast<Index>(A.size()) * sizeof(double);
  eigencuda::checkCuda(cudaMemcpyAsync(this->data(), A.data(), size_A,
                            cudaMemcpyHostToDevice, _stream));
}

CudaMatrix::Unique_ptr_to_GPU_data CudaMatrix::alloc_matrix_in_gpu(
    size_t size_arr) const {
  double *dmatrix;
  throw_if_not_enough_memory_in_gpu(size_arr);
  eigencuda::checkCuda(cudaMalloc(&dmatrix, size_arr));
  Unique_ptr_to_GPU_data dev_ptr(dmatrix,
                                 [](double *x) { eigencuda::checkCuda(cudaFree(x)); });
  return dev_ptr;
}

void CudaMatrix::throw_if_not_enough_memory_in_gpu(
    size_t requested_memory) const {
  size_t free, total;
  eigencuda::checkCuda(cudaMemGetInfo(&free, &total));

  std::ostringstream oss;
  oss << "There were requested : " << requested_memory
      << "bytes Index the device\n";
  oss << "Device Free memory (bytes): " << free
      << "\nDevice total Memory (bytes): " << total << "\n";

  // Raise an error if there is not enough total or free memory in the device
  if (requested_memory > free) {
    oss << "There is not enough memory in the Device!\n";
    throw std::runtime_error(oss.str());
  }
}

}  // namespace eigencuda
