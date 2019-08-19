#include "eigencuda.hpp"
#include <iostream>

namespace eigencuda {

/*
 * Stack a vector of matrices as a single matrix, where each row corresponds
 * to a matrix.
 */
template <typename T> Mat<T> stack(const std::vector<Mat<T>> &tensor) {

  int rows = tensor.size();
  int cols = tensor[0].size(); // size of each matrix

  Mat<T> rs = Mat<T>::Zero(rows, cols);

  for (unsigned i = 0; i < tensor.size(); i++) {
    rs.row(i) = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
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
  // deallocated remaining memory
  for (auto &p : _allocated)
    this->gpu_free(p.second);
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
 * Release the memory associated with the pointer `id` and removed the pointer
 * from the tracked pointers collection
 */
template <typename T> void EigenCuda<T>::free_matrix(int id) {
  // Free Array with id from the device
  gpu_free(_allocated.at(id));
  _allocated.erase(id);
}

/*
 * Deallocate the memory associated with a set of matrices identified
 * by a vector of ints.
 */
// template <typename T>
// void EigenCuda<T>::free_tensor(T *tensor[]) {
//   for (T *x = tensor[]; x != std::end(tensor); ++x) {
//     gpu_free(x);
//   }
// }

/*
 * Allocate memory in the device for matrix A, then if if `copy_to_device`
 * copy the array to the device. Sometimes it only neccesary to allocate
 * space in the device without copying the array because the initial
 * values may not be important like a temporal matrix.
 */
template <typename T>
int EigenCuda<T>::initialize_Matrix(const Mat<T> &A, bool copy_to_device) {

  // size of the Matrices
  std::size_t size_A = A.size() * sizeof(T);

  // Pointer in the device
  T *dA;

  // Allocate either pageable or pinned memory
  gpu_alloc(&dA, size_A);

  // Track the allocated variables
  int id = _counter;
  _allocated.emplace(std::make_pair(id, dA));
  _counter += 1;

  // Transfer data to the GPU
  if (copy_to_device) {
    // Pointers at the host
    const T *hA = A.data();
    cudaMemcpyAsync(dA, hA, size_A, cudaMemcpyHostToDevice, _stream);
  }

  return id;
}

/*
 * Allocate memory in the device for matrix A, then if if `copy_to_device`
 * copy the array to the device. Sometimes it only neccesary to allocate
 * space in the device without copying the array because the initial
 * values may not be important like a temporal matrix.
 */
template <typename T>
T* EigenCuda<T>::initialize_matrix_mem(const Mat<T> &A, bool copy_to_device) {

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
    // cudaMemcpyAsync(dA, hA, size_A, cudaMemcpyHostToDevice, _stream);
    cudaError_t err = cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
    std::cout << "error copying to device: " << err << "\n";
    err = cudaDeviceSynchronize();
    std::cout << "sync err cpy to device: " << err << "\n";

  }

  return dA;
}


/*
 * Retrieve the pointers to the allocated memory in the device using
 * their correspoding ids.
 */
template <typename T>
std::tuple<T *, T *, T *>
EigenCuda<T>::get_pointer_from_ids(std::tuple<int, int, int> ids) {
  // identifiers of the pointers
  int id_A, id_B, id_C;
  std::tie(id_A, id_B, id_C) = ids;

  // Pointer to the arrays in the device
  T *dA, *dB, *dC;
  dA = _allocated.at(id_A);
  dB = _allocated.at(id_B);
  dC = _allocated.at(id_C);

  return std::forward_as_tuple(dA, dB, dC);
}

/*
 * Call the gemm function from cublas, resulting in the multiplication of the
 * two matrices with identifiers id_A and id_B. The result is stored in
 * a Matrix (pointer) with identifier id_C.
 */
template <typename T>
void EigenCuda<T>::gemm(Shapes sh, std::tuple<int, int, int> ids) {
  // get pointers to the device
  T *dA, *dB, *dC;
  std::tie(dA, dB, dC) = get_pointer_from_ids(ids);

  // Scalar constanst for calling blas
  T _alpha = 1.;
  T _beta = 0.;
  const T *_pa = &_alpha;
  const T *_pb = &_beta;

  // call gemm from cublas
  if constexpr (std::is_same<float, T>()) {
      cublasSgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols,
			   sh.A_cols, _pa, dA, sh.A_rows, dB, sh.B_rows, _pb, dC,
			   sh.C_rows);
    } else if (std::is_same<double, T>()) {
    cublasDgemm(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols,
			 sh.A_cols, _pa, dA, sh.A_rows, dB, sh.B_rows, _pb, dC,
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
void EigenCuda<T>::gemmBatched(Shapes sh, T **dA, T **dB, T **dC, int batchCount) {

  // Scalar constanst for calling blas
  T _alpha = 1.;
  T _beta = 0.;
  const T *_pa = &_alpha;
  const T *_pb = &_beta;

  cudaError_t err = cudaDeviceSynchronize();
  std::cout << "Pre sync err: " << err << "\n";

   // call gemm from cublas
  cublasStatus_t status;
  if constexpr (std::is_same<float, T>()) {
      status = cublasSgemmBatched(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols,
				  sh.A_cols, _pa, dA, sh.A_rows, dB, sh.B_rows, _pb, dC,
				  sh.C_rows, batchCount);
  } else if (std::is_same<double, T>()) {
    status = cublasDgemmBatched(_handle, CUBLAS_OP_N, CUBLAS_OP_N, sh.A_rows, sh.B_cols,
				sh.A_cols, _pa, dA, sh.A_rows, dB, sh.B_rows, _pb, dC,
				sh.C_rows, batchCount);
  }
  std::cout << "cuda status: " << status << "\n";
  err = cudaDeviceSynchronize();
  std::cout << "sync err: " << err << "\n";
}
/*
 * Perform the matrix-matrix multiplication between A and B. First,
 * memory is allocated in the device for both matrices then a third temporal
 * array is allocated in the device that will contain the results. Finally, the
 * memory contains in the temporal result is copy back to the main memory and
 * Free the resources
 */
template <typename T>
Mat<T> EigenCuda<T>::dot(const Mat<T> &A, const Mat<T> &B) {
  // Matrix to store the result
  Mat<T> C = Mat<T>::Zero(A.rows(), B.cols());
  std::size_t size_C = C.size() * sizeof(T);

  // Indices of the matrices on the device
  std::tuple<int, int, int> ids = std::make_tuple(
      initialize_Matrix(A), initialize_Matrix(B), initialize_Matrix(C, false));

  // process on GPU
  Shapes sh{A.rows(), A.cols(), B.rows(), B.cols(), C.cols()};
  gemm(sh, ids);

  // send data back to CPU
  T *hC = C.data();
  T *dC = _allocated[std::get<2>(ids)];
  cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

  // Free the result from the device
  free_matrix(std::get<0>(ids)); // Free A
  free_matrix(std::get<1>(ids)); // Free B
  free_matrix(std::get<2>(ids)); // Free C

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, A.rows(), B.cols());

  return C;
}

template <typename T>
std::vector<Mat<T>>
EigenCuda<T>::triple_tensor_product(const Mat<T> &A, const Mat<T> &C,
                                    const std::vector<Mat<T>> &tensor) {
  // Perform the triple matrix multiplication A * matrix * C, for the vector
  // of matrices given by tensor
  std::vector<Mat<T>> rs(tensor.size());

  // Copy Matrix A and B to the device
  int id_A = initialize_Matrix(A);
  int id_C = initialize_Matrix(C);

  // allocate space in device for the temporal matrices
  size_t size_Y = A.rows() * C.cols() * sizeof(T);
  Mat<T> X = Mat<T>::Zero(A.cols(), C.cols());
  Mat<T> Y = Mat<T>::Zero(A.rows(), C.cols());
  Mat<T> matrix = Mat<T>::Zero(A.cols(), C.rows());

  int id_X = initialize_Matrix(X, false);
  int id_Y = initialize_Matrix(Y, false);
  int id_matrix = initialize_Matrix(matrix, false);

  // Iterate over the tensor Using the previous allocated space in the device
  transform(tensor.begin(), tensor.end(), rs.begin(),
            [this, id_A, id_C, id_X, id_Y, id_matrix, size_Y, &A, &C, &X,
             &Y](const Mat<T> &mtx) {
              assert(A.cols() == mtx.rows());
              assert(mtx.cols() == C.rows());

              // Copy matrix to the device
              T *d_matrix = _allocated.at(id_matrix);
              const T *h_mtx = mtx.data();

              // move temporal matrix to the preallocated space
              std::size_t size_mtx = mtx.size() * sizeof(T);
              cudaMemcpyAsync(d_matrix, h_mtx, size_mtx, cudaMemcpyHostToDevice,
                              _stream);

              // Compute first matrix multiplication
              Shapes sh1{mtx.rows(), mtx.cols(), C.rows(), C.cols(), X.rows()};
              std::tuple<int, int, int> ids =
                  std::make_tuple(id_matrix, id_C, id_X);
              gemm(sh1, ids);

              // compute the second matrix multiplication
              Shapes sh2{A.rows(), A.cols(), X.rows(), X.cols(), Y.rows()};
              ids = std::make_tuple(id_A, id_X, id_Y);
              gemm(sh2, ids);

              // send data back to CPU
              T *hY = Y.data();
              T *dY = this->_allocated[id_Y];
              cudaMemcpyAsync(hY, dY, size_Y, cudaMemcpyDeviceToHost, _stream);
              Y = Eigen::Map<Mat<T>>(hY, A.rows(), C.cols());

              return Y;
            });

  // Free all the allocated arrays from the device
  for (int x : {id_A, id_C, id_X, id_Y, id_matrix}) {
    free_matrix(x);
  }

  return rs;
}

/*
 * Multiply a matrix A by a 3D tensor represented as a vector of matrices.
 * Each iteration perform the operation mtx * A,  where mtx is the ith component
 * of the tensor
 */
template <typename T>
std::vector<Mat<T>>
EigenCuda<T>::right_matrix_tensor(const Mat<T> &B,
                                  const std::vector<Mat<T>> &tensor) {
  // Number of submatrices in the input tensor
  int batchCount = tensor.size();
  std::cout << "batchCount: " << batchCount << "\n";

  // Copy Matrix B to the device
  T *mtxB = initialize_matrix_mem(B);

  // allocate space in device for the temporal matrix
  int rows = tensor[0].rows(); // rows of the submatrices
  int cols = tensor[0].cols(); // cols of the submatrices
  Mat<T> matrix = Mat<T>::Zero(rows, cols);

  // Allocate space and copy to the device the input tensor
  T *dA[batchCount];
  for (auto i = 0; i < batchCount; i++) {
    dA[i] = initialize_matrix_mem(tensor[i]);
  }

  // represent the matrix B as a tensor where all the submatrices are the same
  T *dB[batchCount];
  for (auto i = 0; i < batchCount; i++) {
    dB[i] = mtxB;
  }

  // Allocate space in the device for the output tensor
  T *dC[batchCount];
  Mat<T> output = Mat<T>::Zero(matrix.rows(), B.cols());
  for (auto i = 0; i < batchCount; i++) {
    dC[i] = initialize_matrix_mem(output, false);
  }

  // Call tensor matrix multiplication
  Shapes sh{matrix.rows(), matrix.cols(), B.rows(), B.cols(), matrix.rows()};
  gemmBatched(sh, dA, dB, dC, batchCount);

  // Vector containing the results
  std::vector<Mat<T>> rs(batchCount, Mat<T>::Zero(output.rows(), output.cols()));
  std::size_t size_out = output.size() * sizeof(T);

  // Copy the results back to the device
  // for (auto i = 0; i < batchCount; i++) {
    T *hout = rs[0].data();
    T *dout = dC[0];
    // cudaMemcpyAsync(hout, dout, size_out, cudaMemcpyDeviceToHost, _stream);
    cudaError_t err =  cudaMemcpy(hout, dout, size_out, cudaMemcpyDeviceToHost);
    std::cout << "error copying to host: " << err << "\n";
    rs[0] = Eigen::Map<Mat<T>>(hout, output.rows(), output.cols());;
    std::cout << "output: " << rs[0] << "\n";
// }
  // Deallocate all the memory from the device
  gpu_free(mtxB);
  // free_tensor(dA);
  // free_tensor(dC);

  return rs;
}

// explicit instantiations
template class EigenCuda<float>;
template class EigenCuda<double>;
template Mat<float> stack<float>(const std::vector<Mat<float>> &);
template Mat<double> stack<double>(const std::vector<Mat<double>> &);
} // namespace eigencuda
