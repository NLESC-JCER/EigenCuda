#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cublas_v2.h>
#include <curand.h>
#include <tuple>
#include <unordered_map>
#include <vector>

/**
 * \brief Perform matrix-matrix multiplication in a GPU
 *
 * The `EigenCuda` class handles the allocation and deallocation of arrays on
 * the GPU. Firstly, to perform a matrix multiplication, memory must be
 * allocated in the device to contain the involved matrices. The
 * `initialize_matrix` method firstly allocates memory by calling the
 * `gpu_alloc` method that allocates either pinned or pageable memory, see:
 * https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/ Then the
 * array could be optionally copy to the device. The `initialize_matrix` method
 * internally store the pointer in the device to the allocated array and returns
 * and identifier that represents such pointer. This identifier/pointer
 * (key/value) mechanism allows to reuse already allocated space in the device
 * by bookkeeping the identifiers representing the memory.
 *
 */

namespace eigencuda {

inline cudaError_t checkCuda(cudaError_t result) {
// Check Cuda error
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
  }
#endif
  return result;
}

// Structure with the sizes to call ?GEMM
struct Shapes {
  int A_rows;
  int A_cols;
  int B_rows;
  int B_cols;
  int C_rows;

  Shapes(long int _a_rows, long int _a_cols, long int _b_rows, long int _b_cols,
         long int _c_rows)
      : A_rows{static_cast<int>(_a_rows)}, A_cols{static_cast<int>(_a_cols)},
        B_rows{static_cast<int>(_b_rows)}, B_cols{static_cast<int>(_b_cols)},
        C_rows{static_cast<int>(_c_rows)} {}
};

// col Major for CUDA
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T> class EigenCuda {

public:
  EigenCuda() { cublasCreate(&_handle); }
  EigenCuda(bool pinned) : _pinned{pinned} { cublasCreate(&_handle); }

  // Deallocate both the handler and allocated arrays
  ~EigenCuda();

  // Remove the copy operations
  EigenCuda(const EigenCuda &) = delete;
  EigenCuda &operator=(const EigenCuda &) = delete;

  // Matrix matrix multiplication
  Mat<T> dot(const Mat<T> &A, const Mat<T> &B);

  // Perform the triple matrix multiplication A * matrix * C, for the vector
  // of matrices given by tensor
  std::vector<Mat<T>> triple_tensor_product(const Mat<T> &A, const Mat<T> &C,
                                            const std::vector<Mat<T>> &tensor);

  // Perform a multiplication between a matrix and a tensor
  Mat<T> right_matrix_tensor(const Mat<T> &A,
			     const std::vector<Mat<T>> &tensor);

private:
  // Allocate memory in the device
  void gpu_alloc(T **x, std::size_t n) const;

  // Deallocate memory from the device
  void gpu_free(T *x) const;

  // Allocate memory in the device, optionally copying the array to the GPU
  int initialize_Matrix(const Mat<T> &A, bool copy_to_device = true);

  // Invoke the ?gemm function of cublas
  void gemm(Shapes shapes, std::tuple<int, int, int> ids);

  // Deallocate Matrix identifier `id` from the device
  void free_matrix(int id);

  // Shift the pointers of the allocated tensors in the device
  void shift_pointers_by(const std::vector<int> &pointers, const std::vector<long int> &shifts);
  
  // Cuda variables
  cublasHandle_t _handle;
  bool _pinned = false;

  // Allocation booking
  int _counter = 0;
  std::unordered_map<int, T *> _allocated;
};

// Stack a vector of matrices as a matrix where is row contains a matrix
template <typename T> Mat<T> stack(const std::vector<Mat<T>> &tensor);

} // namespace eigencuda

#endif // EIGENCUDA_H_
