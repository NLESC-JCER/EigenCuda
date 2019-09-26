#include "eigencuda.hpp"

namespace eigencuda {

/*
 * Deallocate memory from the device
 */
void free_mem_in_gpu(double *x) { checkCuda(cudaFree(x)); };

EigenCuda::~EigenCuda() {

  // destroy handle
  cublasDestroy(_handle);
  // destroy stream
  cudaStreamDestroy(_stream);
}

/*
 * Check if the available memory is enough to compute the system
 */
void EigenCuda::check_available_memory_in_gpu(size_t requested_memory) const {
  // Use Unified memory
  size_t *free, *total;
  cudaMallocManaged(&free, sizeof(size_t));
  cudaMallocManaged(&total, sizeof(size_t));
  checkCuda(cudaMemGetInfo(free, total));

  std::ostringstream oss;
  oss << "There were requested : " << requested_memory
      << "bytes int the device\n";
  oss << "Device Free memory (bytes): " << *free
      << "\nDevice total Memory (bytes): " << *total << "\n";

  // Raise an error if there is not enough total or free memory in the device
  if (requested_memory > *free) {
    oss << "There is not enough memory in the Device!\n";
    throw std::runtime_error(oss.str());
  }

  // Release memory
  cudaFree(free);
  cudaFree(total);
}

/*
 * Allocate memory in the device for matrix A.
 */
uniq_double EigenCuda::alloc_matrix_in_gpu(size_t size_matrix) const {

  // Pointer in the device
  double *dmatrix;
  checkCuda(cudaMalloc(&dmatrix, size_matrix));
  uniq_double dev_ptr(dmatrix, free_mem_in_gpu);
  return dev_ptr;
}

uniq_double EigenCuda::copy_matrix_to_gpu(const Eigen::MatrixXd &matrix) const {
  // allocate memory in the device
  size_t size_matrix = matrix.size() * sizeof(double);
  uniq_double dev_ptr = alloc_matrix_in_gpu(size_matrix);

  // Transfer data to the GPU
  const double *hmatrix = matrix.data();  // Pointers at the host
  cudaError_t err = cudaMemcpyAsync(dev_ptr.get(), hmatrix, size_matrix,
                                    cudaMemcpyHostToDevice, _stream);
  if (err != 0) {
    throw std::runtime_error("Error copy arrays to device");
  }
  return dev_ptr;
}

/*
 * Call the gemm function from cublas, resulting in the multiplication of the
 * two matrices.
 */
void EigenCuda::gemm(const CudaMatrix &A, const CudaMatrix &B,
                     CudaMatrix &C) const {

  // Scalar constanst for calling blas
  double alpha = 1.;
  double beta = 0.;
  const double *palpha = &alpha;
  const double *pbeta = &beta;

  cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
              palpha, A.ptr(), A.rows(), B.ptr(), B.rows(), pbeta, C.ptr(),
              C.rows());
}

/*
 * Matrix matrix multiplication
 */
Eigen::MatrixXd EigenCuda::matrix_mult(const Eigen::MatrixXd &A,
                                       const Eigen::MatrixXd &B) const {

  // sizes of the matrices to allocated in the device
  size_t size_A = A.size() * sizeof(double);
  size_t size_B = B.size() * sizeof(double);
  size_t size_C = A.rows() * B.cols() * sizeof(double);
  check_available_memory_in_gpu(size_A + size_B + size_C);

  // Send matrices to Nvidia device
  uniq_double dA = copy_matrix_to_gpu(A);
  uniq_double dB = copy_matrix_to_gpu(B);
  uniq_double dC = alloc_matrix_in_gpu(size_C);

  CudaMatrix matrixA{std::move(dA), A.rows(), A.cols()};
  CudaMatrix matrixB{std::move(dB), B.rows(), B.cols()};
  CudaMatrix matrixC{std::move(dC), A.rows(), B.cols()};

  // matrix multiplication
  gemm(matrixA, matrixB, matrixC);

  // Copy the result to the host
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(A.rows(), B.cols());
  double *hout = result.data();
  checkCuda(cudaMemcpyAsync(hout, matrixC.ptr(), size_C, cudaMemcpyDeviceToHost,
                            _stream));

  return result;
}

/*
 * \brief Perform a Tensor3D matrix multiplication
 */
void EigenCuda::right_matrix_tensor_mult(std::vector<Eigen::MatrixXd> &&tensor,
                                         const Eigen::MatrixXd &B) const {
  int batchCount = tensor.size();

  // First submatrix from the tensor
  const Eigen::MatrixXd &submatrix = tensor[0];

  // sizes of the matrices to allocated in the device
  size_t size_A = submatrix.size() * sizeof(double);
  size_t size_B = B.size() * sizeof(double);
  size_t size_C = submatrix.rows() * B.cols() * sizeof(double);
  check_available_memory_in_gpu(size_A + size_B + size_C);

  // Matrix in the Cuda device
  uniq_double dA = alloc_matrix_in_gpu(size_A);
  uniq_double dC = alloc_matrix_in_gpu(size_C);
  uniq_double dB = copy_matrix_to_gpu(B);
  CudaMatrix matrixA{std::move(dA), submatrix.rows(), submatrix.cols()};
  CudaMatrix matrixB{std::move(dB), B.rows(), B.cols()};
  CudaMatrix matrixC{std::move(dC), submatrix.rows(), B.cols()};

  std::vector<Eigen::MatrixXd> result(
      batchCount, Eigen::MatrixXd::Zero(submatrix.rows(), B.cols()));

  // Call tensor matrix multiplication
  for (auto i = 0; i < batchCount; i++) {
    // Copy tensor component to the device
    checkCuda(cudaMemcpyAsync(matrixA.ptr(), tensor[i].data(), size_C,
                              cudaMemcpyHostToDevice, _stream));

    // matrix multiplication
    gemm(matrixA, matrixB, matrixC);

    // Copy the result to the host
    // double *hout = result[i].data();
    double *hout = tensor[i].data();
    checkCuda(cudaMemcpyAsync(hout, matrixC.ptr(), size_C,
                              cudaMemcpyDeviceToHost, _stream));
  }
}

}  // namespace eigencuda
