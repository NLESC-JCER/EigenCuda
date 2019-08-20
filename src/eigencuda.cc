#include "eigencuda.hpp"

namespace eigencuda {

/*
 * Stack a vector of matrices as a single matrix, where each column corresponds
 * to a matrix.
 */
template <typename T> Mat<T> stack(std::vector<Mat<T>> &&tensor) {

  int rows = tensor[0].size(); // size of each matrix
  int cols = tensor.size();    // number of matrices in tensor

  // row major to save the tensor
  Mat<T> rs = Mat<T>::Zero(rows, cols);

  for (auto i = 0; i < cols; i++) {
    rs.col(i) = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        tensor[i].data(), tensor[i].size());
  }
  return rs;
}

/*
 * Removed all the allocated arrays from the device
 */
template <typename T> EigenCuda<T>::~EigenCuda() {
  // destroy handle
  cublasDestroy(_handle);
  // destroy stream
  cudaStreamDestroy(_stream);
}

/*
 * Allocate memory in the device using either pinned or pageable (default)
 * memory
 */
template <typename T> void EigenCuda<T>::gpu_alloc(T **x, std::size_t n) const {
  (_pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
}

/*
 * Deallocate memory from the device
 */
template <typename T> void EigenCuda<T>::gpu_free(T *x) const {
  (_pinned) ? cudaFreeHost(x) : cudaFree(x);
};

/*
 * Allocate memory in the device for matrix A, then if if `copy_to_device`
 * copy the array to the device. Sometimes it only neccesary to allocate
 * space in the device without copying the array because the initial
 * values may not be important like a temporal matrix.
 */
template <typename T>
T *EigenCuda<T>::initialize_matrix_mem(const Mat<T> &A, bool copy_to_device) {

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
 * two matrices with identifiers id_A and id_B. The result is stored in
 * a Matrix (pointer) with identifier id_C.
 */
template <typename T>
void EigenCuda<T>::gemm(Shapes sh, const T *dA, const T *dB, T *dC) {

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
                               int batchCount) {

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
 * Call the gemm?StridedBatched function from cublas, resembling a zip
 * operation of two 3D tensors, applying the gemm function in each pair
 * of matrices. Resulting in a 3D tensor containing the results of the
 * multiplication.
 */
template <typename T>
void EigenCuda<T>::gemmStridedBatched(Shapes sh, Strides st, const T *dA,
                                      const T *dB, T *dC, int batchCount) {

  // call gemm from cublas
  cublasStatus_t status;
  if constexpr (std::is_same<float, T>()) {
    status = cublasSgemmStridedBatched(
        _handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols, sh.A_cols,
        _palpha, dA, sh.A_rows, st.stA, dB, sh.B_rows, st.stB, _pbeta, dC,
        sh.C_rows, st.stC, batchCount);
  } else if (std::is_same<double, T>()) {
    status = cublasDgemmStridedBatched(
        _handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols, sh.A_cols,
        _palpha, dA, sh.A_rows, st.stA, dB, sh.B_rows, st.stB, _pbeta, dC,
        sh.C_rows, st.stC, batchCount);
  }
  if (status != 0) {
    std::runtime_error("error calling cublas?DgemmBatched");
  }
}

/*
 * \brief Matrix-Matrix multiplication in GPU
 */
template <typename T>
Mat<T> EigenCuda<T>::dot(const Mat<T> &A, const Mat<T> &B) {
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
//   // Number of submatrices in the input tensor
//   int batchCount = tensor.size();

//   // Copy Matrix A and C to the device
//   T *mtxA = initialize_matrix_mem(A);
//   T *mtxC = initialize_matrix_mem(C);

//   // allocate space in device for the tensor
//   int rows = tensor[0].rows(); // rows of the submatrices
//   int cols = tensor[0].cols(); // cols of the submatrices
//   Mat<T> matrix = Mat<T>::Zero(rows, cols);

//   // Allocate space and copy to the device the input tensor
//   // Notice that hA, hB and hC are arrays IN THE HOST by the pointers
//   // are allocated in the DEVICE.
//   T *hA[batchCount];
//   for (auto i = 0; i < batchCount; i++) {
//     hA[i] = mtxA; // Use the same pointer
//   }

//   // This array contains the pointers to the allocated space for the tensor
//   T *hB[batchCount];
//   for (auto i = 0; i < batchCount; i++) {
//     hB[i] = initialize_matrix_mem(tensor[i]);
//   }

//   // Use an array of pointers to the same matrix allocated in the device
//   T *hC[batchCount];
//   for (auto i = 0; i < batchCount; i++) {
//     hC[i] = mtxC;
//   }

//   // Allocated space for the temporal matrix
//   // matrix containing the product tensor(i) * C
//   Mat<T> X = Mat<T>::Zero(matrix.rows(), C.cols());
//   T *hX[batchCount];
//   for (auto i = 0; i < batchCount; i++) {
//     hX[i] = initialize_matrix_mem(X, false);
//   }
//   Mat<T> Y = Mat<T>::Zero(A.rows(), C.cols());
//   T *hY[batchCount];
//   for (auto i = 0; i < batchCount; i++) {
//     hY[i] = initialize_matrix_mem(Y, false);
//   }

//   // Allocate space in the device for the array of pointers
//   const T **dA, **dB, **dC, **dZ;
//   T **dX, **dY;
//   size_t size_batch = batchCount * sizeof(T*);
//   cudaMalloc(&dA, size_batch);
//   cudaMalloc(&dB, size_batch);
//   cudaMalloc(&dC, size_batch);
//   cudaMalloc(&dX, size_batch);
//   cudaMalloc(&dY, size_batch);

//   // Copy the arrays of pointers from host to the device
//   cudaMemcpy(dA, hA, size_batch, cudaMemcpyHostToDevice);
//   cudaMemcpy(dB, hB, size_batch, cudaMemcpyHostToDevice);
//   cudaMemcpy(dC, hC, size_batch, cudaMemcpyHostToDevice);
//   cudaMemcpy(dX, hX, size_batch, cudaMemcpyHostToDevice);
//   cudaMemcpy(dY, hY, size_batch, cudaMemcpyHostToDevice);

//   // First tensor matrix multiplication
//   Shapes sh{matrix.rows(), matrix.cols(), C.rows(), C.cols(), matrix.rows()};
//   gemmBatched(sh, dB, dC, dX, batchCount);

//   // Seconds tensor matrix multiplication
//   sh = Shapes{A.rows(), A.cols(), X.rows(), X.cols(), A.rows()};
//   gemmBatched(sh, dA, dX, dY, batchCount);

//   std::vector<Mat<T>> rs;

//   return rs;
// }

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
std::vector<Mat<T>>
EigenCuda<T>::right_matrix_tensor(const Mat<T> &B,
                                  const std::vector<Mat<T>> &tensor) {
  // Number of submatrices in the input tensor
  int batchCount = tensor.size();

  // Copy Matrix B to the device
  T *mtxB = initialize_matrix_mem(B);

  // allocate space in device for the temporal matrix
  int rows = tensor[0].rows(); // rows of the submatrices
  int cols = tensor[0].cols(); // cols of the submatrices
  Mat<T> matrix = Mat<T>::Zero(rows, cols);

  // Allocate space and copy to the device the input tensor
  // Notice that hA, hB and hC are arrays IN THE HOST by the pointers
  // are allocated in the DEVICE.
  T *hA[batchCount];
  for (auto i = 0; i < batchCount; i++) {
    hA[i] = initialize_matrix_mem(tensor[i]);
  }

  // represent the matrix B as a tensor where all the submatrices are the same
  T *hB[batchCount];
  for (auto i = 0; i < batchCount; i++) {
    hB[i] = mtxB;
  }

  // Allocate space in the device for the output tensor
  T *hC[batchCount];
  Mat<T> output = Mat<T>::Zero(matrix.rows(), B.cols());
  for (auto i = 0; i < batchCount; i++) {
    hC[i] = initialize_matrix_mem(output, false);
  }

  // Allocate space in the device for the array of pointers
  const T **dA, **dB;
  T **dC;
  size_t size_batch = batchCount * sizeof(T *);
  cudaMalloc(&dA, size_batch);
  cudaMalloc(&dB, size_batch);
  cudaMalloc(&dC, size_batch);

  // Copy the arrays of pointers from host to the device
  cudaMemcpyAsync(dA, hA, size_batch, cudaMemcpyHostToDevice, _stream);
  cudaMemcpyAsync(dB, hB, size_batch, cudaMemcpyHostToDevice, _stream);
  cudaMemcpyAsync(dC, hC, size_batch, cudaMemcpyHostToDevice, _stream);

  // Call tensor matrix multiplication
  Shapes sh{matrix.rows(), matrix.cols(), B.rows(), B.cols(), matrix.rows()};
  gemmBatched(sh, dA, dB, dC, batchCount);

  // Vector containing the results
  std::vector<Mat<T>> rs(batchCount,
                         Mat<T>::Zero(output.rows(), output.cols()));
  std::size_t size_out = output.size() * sizeof(T);

  // Copy Array of pointers on the device to the host
  cudaMemcpyAsync(hC, dC, size_batch, cudaMemcpyDeviceToHost, _stream);

  // Copy each array back to the device
  for (auto i = 0; i < batchCount; i++) {
    T *hout = rs[i].data();
    T *dout = hC[i];
    cudaMemcpyAsync(hout, dout, size_out, cudaMemcpyDeviceToHost, _stream);
    rs[i] = Eigen::Map<Mat<T>>(hout, output.rows(), output.cols());
    ;
  }
  // Deallocate all the memory from the device
  gpu_free(mtxB);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  for (auto i = 0; i < batchCount; i++) {
    gpu_free(hA[i]);
    gpu_free(hC[i]);
  }

  return rs;
}

/*
 * \brief Multiply a matrix B by a 3D tensor represented as a vector of
 * matrices.
 \return a matrix where each column represent the result product.
 * Initially, it allocates memory and copy the matrix B and the tensor to the
 device.
 *  Also, the function allocates the result tensor Y.
 * The method iterates over each submatrix of the tensor computing:
 * output(i) = tensor(i) * A.
 * Finally, the tensor output is copy back to the main memory.
 */
template <typename T>
Mat<T> EigenCuda<T>::matrix_tensor(const Mat<T> &B,
                                   std::vector<Mat<T>> &&tensor) {
  // Number of submatrices in the input tensor
  int batchCount = tensor.size();

  // Rows and cols of the submatrices in tensor
  long int rows_matrix = tensor[0].rows();
  long int cols_matrix = tensor[0].cols();

  // Copy Matrix B to the device
  const T *dB = initialize_matrix_mem(B);

  // Stack the input vector
  Mat<T> super_matrix = stack(std::move(tensor));

  // Allocate memory in the device for the super matrix
  const T *dA = initialize_matrix_mem(super_matrix);

  // Allocate memory in the device for the result
  T *dC;
  long int dim_out = rows_matrix * B.cols();
  size_t size_out = batchCount * static_cast<int>(dim_out) * sizeof(T);
  gpu_alloc(&dC, size_out);

  // Call the matrix multiplication
  Shapes sh{rows_matrix, cols_matrix, B.rows(), B.cols(), rows_matrix};
  Strides st{rows_matrix * cols_matrix, 0, dim_out};
  gemmStridedBatched(sh, st, dA, dB, dC, batchCount);

  // Vector containing the results
  int rows = static_cast<int>(dim_out);
  Mat<T> output = Mat<T>::Zero(rows, batchCount);

  // copy array back to the host
  T *hout = output.data();
  cudaMemcpyAsync(hout, dC, size_out, cudaMemcpyDeviceToHost, _stream);
  output = Eigen::Map<Mat<T>>(hout, rows, batchCount);

  return output;
}

// explicit instantiations
template class EigenCuda<float>;
template class EigenCuda<double>;
} // namespace eigencuda
