#include "eigencuda.hpp"

namespace eigencuda {

  
template <typename T>
std::vector<Mat<T>>
TensorMatrix<T>::tensor_dot_matrix() {
  std::vector<Mat<T>> rs;
  return rs;
}

  
  
// explicit instantiations
template class TensorMatrix<float>;
template class TensorMatrix<double>;
} // namespace eigencuda
