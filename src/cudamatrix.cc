#include "cudamatrix.hpp"

namespace eigencuda {

CudaMatrix::CudaMatrix(const Eigen::MatrixXd &matrix,
                       const cudaStream_t &stream)
    : _rows{static_cast<Index>(matrix.rows())},
      _cols{static_cast<Index>(matrix.cols())} {
  _data = this->alloc_tensor_in_gpu(size_tensor());
  _stream = stream;
  cudaError_t err = cudaMemcpyAsync(_data.get(), matrix.data(), size_tensor(),
                                    cudaMemcpyHostToDevice, stream);
  if (err != 0) {
    throw std::runtime_error("Error copy arrays to device");
  }
}

CudaMatrix::CudaMatrix(Index nrows, Index ncols, const cudaStream_t &stream)
    : _rows{static_cast<Index>(nrows)}, _cols{static_cast<Index>(ncols)} {
  _data = this->alloc_tensor_in_gpu(size_tensor());
  _stream = stream;
}

CudaMatrix::operator Eigen::MatrixXd() const {
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(this->rows(), this->cols());
  eigencuda::checkCuda(cudaMemcpyAsync(result.data(), this->data(),
                                       this->size_tensor(),
                                       cudaMemcpyDeviceToHost, this->_stream));
  eigencuda::checkCuda(cudaStreamSynchronize(this->_stream));
  return result;
}

void CudaMatrix::copy_to_gpu(const Eigen::MatrixXd &A) {
  size_t size_A = static_cast<Index>(A.size()) * sizeof(double);
  eigencuda::checkCuda(cudaMemcpyAsync(this->data(), A.data(), size_A,
                                       cudaMemcpyHostToDevice, _stream));
}

}  // namespace eigencuda
