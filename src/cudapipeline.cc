
#include "cudapipeline.hpp"

namespace eigencuda {
  CudaPipeline::~CudaPipeline() {

  // destroy handle
  cublasDestroy(_handle);
  // destroy stream
  cudaStreamDestroy(_stream);
}

/*
 * Call the gemm function from cublas, resulting in the multiplication of the
 * two matrices
 */
void CudaPipeline::gemm(const CudaMatrix &A, const CudaMatrix &B,
                        CudaMatrix &C) const {

  // Scalar constanst for calling blas
  double alpha = 1.;
  double beta = 0.;
  const double *palpha = &alpha;
  const double *pbeta = &beta;

  if ((A.cols() != B.rows())) {
    throw std::runtime_error("Shape mismatch in Cublas gemm");
  }
  cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, int(A.rows()), int(B.cols()),
              int(A.cols()), palpha, A.data(), int(A.rows()), B.data(),
              int(B.rows()), pbeta, C.data(), int(C.rows()));
}

}  // namespace eigencuda
