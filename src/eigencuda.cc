#include "eigencuda.hpp"
#include <sstream>

namespace eigencuda {

template <typename T>
EigenCuda<T>::~EigenCuda() {

  // destroy handle
  cublasDestroy(_handle);
  // destroy stream
  cudaStreamDestroy(_stream);
}

/*
 * Allocate memory in the device using either pinned or pageable (default)
 * memory
 */
template <typename T>
void EigenCuda<T>::gpu_alloc(T **x, std::size_t n) const {
  (_pinned) ? checkCuda(cudaMallocHost(x, n)) : checkCuda(cudaMalloc(x, n));
}

/*
 * Deallocate memory from the device
 */
template <typename T>
void EigenCuda<T>::gpu_free(T *x) const {
  (_pinned) ? checkCuda(cudaFreeHost(x)) : checkCuda(cudaFree(x));
};

/*
 * Check if the available memory is enough to compute the system
 */
template <typename T>
void EigenCuda<T>::check_available_memory(size_t requested) const {
  size_t *free, *total;

  // Use Unified memory
  cudaMallocManaged(&free, sizeof(size_t));
  cudaMallocManaged(&total, sizeof(size_t));  
  checkCuda(cudaMemGetInfo(free, total));

  std::ostringstream oss;
  oss << "There were requested : " << requested << "bytes int the device\n";  
  oss << "Device Free memory (bytes): " << *free << "\nDevice total Memory (bytes): "  << *total << "\n";

  // Raise an error if there is not enough total or free memory in the device
  if (requested > *free)  {
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
template <typename T>
T *EigenCuda<T>::initialize_matrix_mem(size_t size_A) const {

  // Pointer in the device
  T *dA;

  // Allocate either pageable or pinned memory
  gpu_alloc(&dA, size_A);

  return dA;
}

/*
 * Allocate memory for the matrix and copy it to the device
 */
template <typename T>
T *EigenCuda<T>::initialize_and_copy(const Mat<T> &A) const {

  // allocate memory in the device
  size_t size_A = A.size() * sizeof(T);
  T *dA = initialize_matrix_mem(size_A);

  // Transfer data to the GPU 
  const T *hA = A.data();   // Pointers at the host
  cudaError_t err =
    cudaMemcpyAsync(dA, hA, size_A, cudaMemcpyHostToDevice, _stream);
  if (err != 0) {
    throw std::runtime_error("Error copy arrays to device");
  }
  return dA;
}
  
  
/*
 * Call the gemm function from cublas, resulting in the multiplication of the
 * two matrices.
 */
template <typename T>
void EigenCuda<T>::gemm(Shapes sh, const T *dA, const T *dB, T *dC) const {

  // Scalar constanst for calling blas
  T _alpha = 1.;
  T _beta = 0.;
  const T *_palpha = &_alpha;
  const T *_pbeta = &_beta;
  
  // call gemm from cublas
  if constexpr (std::is_same<float, T>()) {
    cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols,
                sh.A_cols, _palpha, dA, sh.A_rows, dB, sh.B_rows, _pbeta, dC,
                sh.C_rows);
  } else if (std::is_same<double, T>()) {
    cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols,
                sh.A_cols, _palpha, dA, sh.A_rows, dB, sh.B_rows, _pbeta, dC,
                sh.C_rows);
  }
}

/*
 * \brief Matrix-Matrix multiplication in GPU
 */
template <typename T>
Mat<T> EigenCuda<T>::dot(const Mat<T> &A, const Mat<T> &B) const {
  // Matrix to store the result
  Mat<T> C = Mat<T>::Zero(A.rows(), B.cols());
  std::size_t size_C = C.size() * sizeof(T);

  // Indices of the matrices on the device
  T *dA = initialize_and_copy(A);
  T *dB = initialize_and_copy(B);
  T *dC = initialize_matrix_mem(size_C);

  // process on GPU
  Shapes sh{A.rows(), A.cols(), B.rows(), B.cols(), C.rows()};
  gemm(sh, dA, dB, dC);

  // send data back to CPU
  T *hC = C.data();
  cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

  // Free the result from the device
  gpu_free(dA);
  gpu_free(dB);
  gpu_free(dC);

  return C;
}

/*
 * \brief Multiply a matrix B by a 3D tensor represented as a vector of
 * matrices.
 * \return vector of matrices representing the result
 * Initially, it allocates memory and copy the matrix B and each submatrix
 * from tensor to the device. Also, the function allocates the result matrix
 * C into the device. The method iterates
 * over each submatrix of the tensor computing: C = tensor(i) * A.
 */
template <typename T>
std::vector<Mat<T>> EigenCuda<T>::right_matrix_tensor(
    const Mat<T> &B, const std::vector<Mat<T>> &tensor) const {
  // Number of submatrices in the input tensor
  int batchCount = tensor.size();

  // First submatrix from the tensor
  Mat<T> matrix = tensor[0];

  // sizes of the matrices to allocated in the device
  size_t size_A = matrix.size() * sizeof(T);
  size_t size_B = B.size() * sizeof(T);
  size_t size_C = matrix.rows() * B.cols() * sizeof(T);  

  // Check if there is enough available memory
  check_available_memory(size_A + size_B + size_C);
  
  // Initialize memory for tensor components
  T *dA = initialize_matrix_mem(size_A);

  // Allocate memory for the final result array C
  T *dC = initialize_matrix_mem(size_C);
  
  // Copy Matrix B to the device
  T *dB = initialize_and_copy(B);

  // Shapes of the resulting matrices
  Shapes sh{matrix.rows(), matrix.cols(), B.rows(), B.cols(), matrix.rows()};

  // Vector containing the results
  std::vector<Mat<T>> rs(batchCount, Mat<T>::Zero(matrix.rows(), B.cols()));
  
  // Call tensor matrix multiplication
  for (auto i=0; i < batchCount; i++) {
    // Copy tensor component to the device
    checkCuda(cudaMemcpyAsync(dA, tensor[i].data(), size_C, cudaMemcpyHostToDevice, _stream));
    
    // matrix multiplication
    gemm(sh, dA, dB, dC);

    // Copy the result to the host
    T *hout = rs[i].data();
    checkCuda(cudaMemcpyAsync(hout, dC, size_C, cudaMemcpyDeviceToHost, _stream));
  }

  // Deallocate all the memory from the device
  checkCuda(cudaFree(dA));
  checkCuda(cudaFree(dB));
  checkCuda(cudaFree(dC));

  return rs;
}

/*
 * \brief performs a matrix_1 * tensor * matrix_2 multiplication
 * \return vector containging the matrix-matrix multiplications
 */
template <typename T>
std::vector<Mat<T>> EigenCuda<T>::triple_tensor_product(
    const Mat<T> &A, const Mat<T> &C, const std::vector<Mat<T>> &tensor) {
  // Number of submatrices in the input tensor
  int batchCount = tensor.size();

  // First submatrix from the tensor
  Mat<T> matrix = tensor[0];
  
  // sizes of the matrices to allocated in the device
  size_t size_A = A.size() * sizeof(T);
  size_t size_B = matrix.size() * sizeof(T);
  size_t size_C = C.size() * sizeof(T);  
  std::size_t size_X = A.rows() * matrix.cols() * sizeof(T);
  std::size_t size_Y = A.rows() * C.cols() * sizeof(T);

  // Check if there is enough available memory
  check_available_memory(size_A + size_B + size_C + size_X + size_Y);  
  
  // Copy Matrix B to the device
  T *dA = initialize_and_copy(A);
  T *dC = initialize_and_copy(C);
  T *dB = initialize_matrix_mem(size_B);

  // Intermediate result X
  T *dX = initialize_matrix_mem(size_X) ;
  
  // Final result array Y
  T *dY = initialize_matrix_mem(size_Y);

  // Shapes of the matrices
  Shapes sh1{A.rows(), A.cols(), matrix.rows(), matrix.cols(), A.rows()};
  Shapes sh2{A.rows(), matrix.cols(), C.rows(), C.cols(), A.rows()};  

  // Vector containing the results
  std::vector<Mat<T>> rs(batchCount, Mat<T>::Zero(A.rows(), C.cols()));

  for (auto i=0; i < batchCount; i++) {
    // tensor component
    checkCuda(cudaMemcpyAsync(dB, tensor[i].data(), size_B, cudaMemcpyHostToDevice, _stream));

    // Call the first tensor matrix multiplication
    gemm(sh1, dA, dB, dX);

    // Call the second tensor matrix multiplication
    gemm(sh2, dX, dC, dY);

    // Copy the result Array back to the device
    T *hout = rs[i].data();
    checkCuda(cudaMemcpyAsync(hout, dY, size_Y, cudaMemcpyDeviceToHost, _stream));
  }

  // Deallocate all the memory from the device
  checkCuda(cudaFree(dA));
  checkCuda(cudaFree(dB));
  checkCuda(cudaFree(dC));
  checkCuda(cudaFree(dX));
  checkCuda(cudaFree(dY));

  return rs;
}

// explicit instantiations
template class EigenCuda<float>;
template class EigenCuda<double>;
}  // namespace eigencuda
