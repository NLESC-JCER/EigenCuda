#include "eigencuda.hpp"

namespace eigencuda {

/*
 * Removed all the allocated arrays from the device
 */
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
 * Allocate memory in the device for a tensor
 */
template <typename T>
void EigenCuda<T>::gpu_alloc_tensor(T *arr[], int shape, int batchCount) const {
  // size of the submatrix inside the tensor
  size_t size_matrix = shape * sizeof(T);

  for (auto i = 0; i < batchCount; i++) {
    // Pointer in the device
    T *dA;
    // Allocate memory
    gpu_alloc(&dA, size_matrix);
    arr[i] = dA;
  }
}

/*
 * Free the memory allocated for a tensor
 */
template <typename T>
void EigenCuda<T>::free_tensor_memory(T *arr[], int batchCount) const {
  for (auto i = 0; i < batchCount; i++) {
    gpu_free(arr[i]);
  }
}

/*
 * Copy each component of the tensor to preallocated memory in the device
 */
template <typename T>
void EigenCuda<T>::copy_tensor_to_dev(const std::vector<Mat<T>> &tensor,
                                      T *arr[]) const {
  size_t size_A = tensor[0].size() * sizeof(T);

  // Send each matrix one by one
  for (unsigned i = 0; i < tensor.size(); i++) {
    const T *hA = tensor[i].data();
    checkCuda(cudaMemcpyAsync(arr[i], hA, size_A, cudaMemcpyHostToDevice, _stream));
  }
}

/*
 * Allocate memory in the device for matrix A, then if `copy_to_device` is true
 * copy the array to the device. Sometimes it only neccesary to allocate
 * space in the device without copying the array because the initial
 * values may not be important like a temporal matrix.
 */
template <typename T>
T *EigenCuda<T>::initialize_matrix_mem(const Mat<T> &A,
                                       bool copy_to_device) const {

  // size of the Matrices
  std::size_t size_A = A.size() * sizeof(T);

  // Pointer in the device
  T *dA;

  // Allocate either pageable or pinned memory
  gpu_alloc(&dA, size_A);

  // Transfer data to the GPU
  if (copy_to_device) {
    // Pointers at the host
    const T *hA = A.data();
    cudaError_t err =
        cudaMemcpyAsync(dA, hA, size_A, cudaMemcpyHostToDevice, _stream);
    if (err != 0) {
      throw std::runtime_error("Error copy arrays to device");
    }
  }
  return dA;
}

/*
 * Call the gemm function from cublas, resulting in the multiplication of the
 * two matrices.
 */
template <typename T>
void EigenCuda<T>::gemm(Shapes sh, const T *dA, const T *dB, T *dC) const {

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
 * Call the gemm?Batched function from cublas, resembling a zip
 * operation of two 3D tensors, applying the gemm function in each pair
 * of matrices. Resulting in a 3D tensor containing the results of the
 * multiplication.
 */
template <typename T>
void EigenCuda<T>::gemmBatched(Shapes sh, const T **dA, const T **dB, T **dC,
                               int batchCount) const {

  // call gemm from cublas
  cublasStatus_t status;
  if constexpr (std::is_same<float, T>()) {
    status =
        cublasSgemmBatched(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows,
                           sh.B_cols, sh.A_cols, _palpha, dA, sh.A_rows, dB,
                           sh.B_rows, _pbeta, dC, sh.C_rows, batchCount);
  } else if (std::is_same<double, T>()) {
    status =
        cublasDgemmBatched(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows,
                           sh.B_cols, sh.A_cols, _palpha, dA, sh.A_rows, dB,
                           sh.B_rows, _pbeta, dC, sh.C_rows, batchCount);
  }
  if (status != 0) {
    std::runtime_error("error calling cublas?DgemmBatched");
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
  T *dA = initialize_matrix_mem(A);
  T *dB = initialize_matrix_mem(B);
  T *dC = initialize_matrix_mem(C, false);

  // process on GPU
  Shapes sh{A.rows(), A.cols(), B.rows(), B.cols(), C.rows()};
  gemm(sh, dA, dB, dC);

  // send data back to CPU
  T *hC = C.data();
  cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, A.rows(), B.cols());

  // Free the result from the device
  gpu_free(dA);
  gpu_free(dB);
  gpu_free(dC);

  return C;
}

/*
 * \brief performs a matrix_1 * tensor * matrix_2 multiplication
 * \return matrix where each column is the result of the matrices
 * multiplication.
 *
 * Initially, it allocates memory and copy the matrices A and C together with
 * the tensor to the device. Also, the function allocates the result tensor Y
 * and a temporal matrix X.
 * This last matrix is not copy into the device because is initial value is not
 * relevant. Subsequently, the method iterates over each submatrix in `tensor`
 * and perform the following operations: X = tensor(i) * C Y(i) = A * X then the
 * final Y is copy back to main memory. This final matrix Y contains in each
 * column the result of the tensor operation. Also, notice that the matrix X is
 * never set to zero after each iteration because the gemm function perform the
 * matrix multiplication: R = alpha M * N + beta R where alpha and beta are two
 * scalar constants set to 1 and 0 respectively. Therefore, X is ALWAYS SET TO
 * ZERO BEFORE THE MATRIX MULTIPLICATION.
 */
// template <typename T>
// std::vector<Mat<T>>
// EigenCuda<T>::triple_tensor_product(const Mat<T> &A, const Mat<T> &C,
//                                     const std::vector<Mat<T>> &tensor) {

/*
 * \brief Multiply a matrix B by a 3D tensor represented as a vector of
 * matrices.
 * \return vector of matrices representing the result
 * Initially, it allocates memory and copy the matrix B and the tensor to the
 * device. Also, the function allocates the result tensor Y. The method iterates
 * over each submatrix of the tensor computing: output(i) = tensor(i) * A.
 * Finally, the tensor output is copy back to the main memory.
 */
template <typename T>
std::vector<Mat<T>> EigenCuda<T>::right_matrix_tensor(
    const Mat<T> &B, const std::vector<Mat<T>> &tensor) const {
  // Number of submatrices in the input tensor
  int batchCount = tensor.size();

  // Copy Matrix B to the device
  T *mtxB = initialize_matrix_mem(B);

  // First submatrix from the tensor
  Mat<T> matrix = tensor[0];

  // Allocate space and copy to the device the input tensor
  // Notice that hA, hB and hC are arrays IN THE HOST by the pointers
  // are allocated in the DEVICE.
  T *hA[batchCount];
  gpu_alloc_tensor(hA, matrix.size(), batchCount);
  copy_tensor_to_dev(tensor, hA);

  // represent the matrix B as a tensor where all the submatrices are the same
  T *hB[batchCount];
  for (auto i = 0; i < batchCount; i++) {
    hB[i] = mtxB;
  }

  // Allocate space in the device for the output tensor
  T *hC[batchCount];
  gpu_alloc_tensor(hC, matrix.rows() * B.cols(), batchCount);

  // Allocate space in the device for the array of pointers
  const T **dA, **dB;
  T **dC;
  size_t size_batch = batchCount * sizeof(T *);
  checkCuda(cudaMalloc(&dA, size_batch));
  checkCuda(cudaMalloc(&dB, size_batch));
  checkCuda(cudaMalloc(&dC, size_batch));

  // Copy the arrays of pointers from host to the device
  checkCuda(cudaMemcpyAsync(dA, hA, size_batch, cudaMemcpyHostToDevice, _stream));
  checkCuda(cudaMemcpyAsync(dB, hB, size_batch, cudaMemcpyHostToDevice, _stream));
  checkCuda(cudaMemcpyAsync(dC, hC, size_batch, cudaMemcpyHostToDevice, _stream));

  // Call tensor matrix multiplication
  Shapes sh{matrix.rows(), matrix.cols(), B.rows(), B.cols(), matrix.rows()};
  gemmBatched(sh, dA, dB, dC, batchCount);

  // Vector containing the results
  std::vector<Mat<T>> rs(batchCount, Mat<T>::Zero(matrix.rows(), B.cols()));
  std::size_t size_out = matrix.rows() * B.cols() * sizeof(T);

  // Copy Array of pointers on the device to the host
  checkCuda(cudaMemcpyAsync(hC, dC, size_batch, cudaMemcpyDeviceToHost, _stream));

  // Copy each array back to the device
  for (auto i = 0; i < batchCount; i++) {
    T *hout = rs[i].data();
    T *dout = hC[i];
    checkCuda(cudaMemcpyAsync(hout, dout, size_out, cudaMemcpyDeviceToHost, _stream));
    rs[i] = Eigen::Map<Mat<T>>(hout, matrix.rows(), B.cols());
  }
  // Deallocate all the memory from the device
  gpu_free(mtxB);
  checkCuda(cudaFree(dA));
  checkCuda(cudaFree(dB));
  checkCuda(cudaFree(dC));
  free_tensor_memory(hA, batchCount);
  free_tensor_memory(hC, batchCount);

  return rs;
}

// explicit instantiations
template class EigenCuda<float>;
template class EigenCuda<double>;
}  // namespace eigencuda
