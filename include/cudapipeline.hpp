#ifndef CUDA_PIPELINE__H
#define CUDA_PIPELINE__H

#include "eigencuda.hpp"

/*
 * \brief Perform Tensor-matrix multiplications in a GPU
 *
 * The `CudaPipeline` class handles the allocation and deallocation of arrays on
 * the GPU.
 */

namespace eigencuda {

/* \brief The CudaPipeline class offload Eigen operations to an *Nvidia* GPU
 * using the CUDA language. The Cublas handle is the context manager for all the
 * resources needed by Cublas. While a stream is a queue of sequential
 * operations executed in the Nvidia device.
 */
class CudaPipeline {
 public:
  CudaPipeline() {
    cublasCreate(&_handle);
    cudaStreamCreate(&_stream);
  }
  ~CudaPipeline();

  CudaPipeline(const CudaPipeline &) = delete;
  CudaPipeline &operator=(const CudaPipeline &) = delete;

  // Invoke the ?gemm function of cublas
  void gemm(const CudaMatrix &A, const CudaMatrix &B, CudaMatrix &C) const;

  const cudaStream_t &get_stream() const { return _stream; };

 private:
  // The cublas handles allocates hardware resources on the host and device.
  cublasHandle_t _handle;

  // Asynchronous stream
  cudaStream_t _stream;
};

}  // namespace eigencuda

#endif
