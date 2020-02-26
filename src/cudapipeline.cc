
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
  cublasSetStream(_handle, _stream);
  cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, int(A.rows()), int(B.cols()),
              int(A.cols()), palpha, A.data(), int(A.rows()), B.data(),
              int(B.rows()), pbeta, C.data(), int(C.rows()));
}

/**
 * \brief Call the gemmbatch function from cublas to perform tensor tensor
 * mutlplications
 * @param A first tensor
 * @param B seconds tensor
 * @param C result
 */
void CudaPipeline::gemmbatch(const CudaTensor &A, const CudaTensor &B,
                             CudaTensor &C) const {
  // cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
  //                                   cublasOperation_t transa,
  //                                   cublasOperation_t transb,
  //                                   int m, int n, int k,
  //                                   const double          *alpha,
  //                                   const double          *Aarray[], int lda,
  //                                   const double          *Barray[], int ldb,
  //                                   const double          *beta,
  //                                   double          *Carray[], int ldc,
  //                                   int batchCount)
}

}  // namespace eigencuda
