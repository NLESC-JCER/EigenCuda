#include "cudamatrix.hpp"
#include "cudapipeline.hpp"
#include <algorithm>
#include <omp.h>

using eigencuda::CudaMatrix;
using eigencuda::CudaPipeline;
using eigencuda::Index;

int main() {
  Index dim = 2000;
  Index nvectors = 10;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(dim, dim);

  CudaPipeline cuda_pip, cuda_pip2;
  CudaMatrix cuma_A{A, cuda_pip.get_stream()};
  CudaMatrix cuma_Z{A, cuda_pip2.get_stream()};
  CudaMatrix cuma_B{dim, dim, cuda_pip.get_stream()};
  CudaMatrix cuma_C{dim, dim, cuda_pip.get_stream()};
  CudaMatrix cuma_X{dim, dim, cuda_pip2.get_stream()};
  CudaMatrix cuma_Y{dim, dim, cuda_pip2.get_stream()};

  std::vector<Eigen::MatrixXd> tensor(nvectors);
  std::vector<Eigen::MatrixXd> results(nvectors,
                                       Eigen::MatrixXd::Zero(dim, dim));
  std::for_each(tensor.begin(), tensor.end(),
                [dim](auto& x) { return Eigen::MatrixXd::Random(dim, dim); });

  omp_set_num_threads(2);
#pragma omp parallel for schedule(dynamic)
  for (Index i = 0; i < tensor.size(); i++) {
    auto id = omp_get_thread_num();
    if ((id % 2) == 0) {
      cuma_B.copy_to_gpu(tensor[i]);
      cuda_pip.gemm(cuma_B, cuma_A, cuma_C);
      results[i] = cuma_C;
    } else {
      cuma_X.copy_to_gpu(tensor[i]);
      cuda_pip2.gemm(cuma_X, cuma_Z, cuma_Y);
      results[i] = cuma_Y;
    }
  }
  for (Index i = 0; i < tensor.size(); i++) {
    std::cout << "results: " << std::boolalpha
              << results[i].isApprox(A * tensor[i]) << "\n";
  }
}
