#include "eigencuda.hpp"

namespace eigencuda {

  
template <typename T>
std::vector<Mat<T>>
MatrixTensor<T>::tensor_dot_matrix() {
  std::vector<Mat<T>> rs;
  return rs;
}

  
  
// explicit instantiations
template class MatrixTensor<float>;
template class MatrixTensor<double>;
} // namespace eigencuda
