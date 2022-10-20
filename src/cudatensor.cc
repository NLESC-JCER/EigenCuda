#include <cudatensor.hpp>
#include <sstream>

namespace eigencuda {
Index CudaTensor::size() const {
  return std::accumulate(this->_dimensions.cbegin(), this->_dimensions.cend(),
                         1, [](auto x, auto y) { return x * y; });
}
void CudaTensor::copy_to_gpu(const Eigen::Tensor<double, 3> &A) {
  size_t size_A = static_cast<Index>(A.size()) * sizeof(double);
  eigencuda::checkCuda(cudaMemcpyAsync(this->data(), A.data(), size_A,
                                       cudaMemcpyHostToDevice, _stream));
}

// CudaTensor::operator Eigen::Tensor<double, 3>() const {
//   Eigen::Tensor<double, 3>::Zero(this->rows(), this->cols());
//   checkCuda(cudaMemcpyAsync(result.data(), this->data(), this->size_matrix(),
//                             cudaMemcpyDeviceToHost, this->_stream));
//   checkCuda(cudaStreamSynchronize(this->_stream));
//   return result;
// }

}  // namespace eigencuda
