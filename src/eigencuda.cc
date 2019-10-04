#include "eigencuda.hpp"

namespace eigencuda {
CudaPipeline::~CudaPipeline() {

  // destroy handle
  cublasDestroy(_handle);
  // destroy stream
  cudaStreamDestroy(_stream);
}

void CudaPipeline::throw_if_not_enough_memory_in_gpu(
    size_t requested_memory) const {
  size_t free, total;
  checkCuda(cudaMemGetInfo(&free, &total));

  std::ostringstream oss;
  oss << "There were requested : " << requested_memory
      << "bytes int the device\n";
  oss << "Device Free memory (bytes): " << free
      << "\nDevice total Memory (bytes): " << total << "\n";

  // Raise an error if there is not enough total or free memory in the device
  if (requested_memory > free) {
    oss << "There is not enough memory in the Device!\n";
    throw std::runtime_error(oss.str());
  }
}

/*
 * Allocate memory in the device for matrix A.
 */
CudaMatrix::double_unique_ptr CudaMatrix::alloc_matrix_in_gpu(
    size_t size_matrix) const {

  // Pointer in the device
  double *dmatrix;
  checkCuda(cudaMalloc(&dmatrix, size_matrix));
  double_unique_ptr dev_ptr(dmatrix, [](double *x) { checkCuda(cudaFree(x)); });
  return dev_ptr;
}

/*
 * Call the gemm function from cublas, resulting in the multiplication of the
 * two matrices.
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

  cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
              palpha, A.pointer(), A.rows(), B.pointer(), B.rows(), pbeta,
              C.pointer(), C.rows());
}

/*
 * \brief Perform a Tensor3D matrix multiplication
 */
void CudaPipeline::right_matrix_tensor_mult(
    std::vector<Eigen::MatrixXd> &tensor, const Eigen::MatrixXd &B) const {
  // First submatrix from the tensor
  const Eigen::MatrixXd &submatrix = tensor[0];

  // sizes of the matrices to allocated in the device
  size_t size_A = submatrix.size() * sizeof(double);
  size_t size_B = B.size() * sizeof(double);
  size_t size_C = submatrix.rows() * B.cols() * sizeof(double);
  throw_if_not_enough_memory_in_gpu(size_A + size_B + size_C);

  // Matrix in the Cuda device

  CudaMatrix matrixA(submatrix.rows(), submatrix.cols());
  CudaMatrix matrixB{B, _stream};
  CudaMatrix matrixC(submatrix.rows(), B.cols());

  // Call tensor matrix multiplication
  for (auto i = 0; i < static_cast<int>(tensor.size()); i++) {
    // Copy tensor component to the device
    checkCuda(cudaMemcpyAsync(matrixA.pointer(), tensor[i].data(), size_C,
                              cudaMemcpyHostToDevice, _stream));

    // matrix multiplication
    gemm(matrixA, matrixB, matrixC);

    // Copy the result to the host
    double *hout = tensor[i].data();
    checkCuda(cudaMemcpyAsync(hout, matrixC.pointer(), size_C,
                              cudaMemcpyDeviceToHost, _stream));
  }
}

/*
 * \brief Perform a Tensor3D matrix multiplication
 */
Eigen::MatrixXd CudaPipeline::matrix_mult(const Eigen::MatrixXd &A,
                                          const Eigen::MatrixXd &B) const {

  // sizes of the matrices to allocated in the device
  size_t size_A = A.size() * sizeof(double);
  size_t size_B = B.size() * sizeof(double);
  size_t size_C = A.rows() * B.cols() * sizeof(double);
  throw_if_not_enough_memory_in_gpu(size_A + size_B + size_C);

  // Matrix in the Cuda device
  CudaMatrix matrixA{A, _stream};
  CudaMatrix matrixB{B, _stream};
  CudaMatrix matrixC{A.rows(), B.cols()};

  // matrix multiplication
  gemm(matrixA, matrixB, matrixC);

  // Copy the result to the host
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(A.rows(), matrixC.cols());
  checkCuda(cudaMemcpyAsync(result.data(), matrixC.pointer(), size_C,
                            cudaMemcpyDeviceToHost, _stream));
  return result;
}

}  // namespace eigencuda
