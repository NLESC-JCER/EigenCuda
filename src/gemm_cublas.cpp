
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <curand.h>
#include <cxxopts.hpp>
#include <iostream>

// Check Cuda error
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << "\n";
  }
#endif
  return result;
}

// col Major for CUDA
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
Mat<T> cublas_gemm(Mat<T> A, Mat<T> B, bool pinned = false) {

  // Scalar constanst for calling blas
  constexpr T alpha = 1.;
  constexpr T beta = 0.;
  const T *pa = &alpha;
  const T *pb = &beta;

  // Size of the Matrices
  std::size_t size_A = A.rows() * A.cols() * sizeof(T);
  std::size_t size_B = A.cols() * B.cols() * sizeof(T);
  std::size_t size_C = A.rows() * B.cols() * sizeof(T);

  Mat<T> C = Mat<T>::Zero(A.rows(), B.cols());

  // and their pointers
  T *hA = A.data();
  T *hB = B.data();
  T *hC = C.data();

  // alloc memory on the GPU
  T *dA, *dB, *dC;

  // Allocate either pageable or pinned memory
  auto fun_alloc = [&](T **x, std::size_t n) {
    (pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
  };

  fun_alloc(&dA, size_A);
  fun_alloc(&dB, size_B);
  fun_alloc(&dC, size_C);

  // cuda handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Transfer data to GPU
  cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);

  // process on GPU
  if constexpr (std::is_same<float, T>::value) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                pa, dA, A.rows(), dB, B.rows(), pb, dC, C.rows());
  } else {
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.rows(), B.cols(), A.cols(),
                pa, dA, A.rows(), dB, B.rows(), pb, dC, C.rows());
  }

  // send data back to CPU
  cudaMemcpy(hC, dC, size_C, cudaMemcpyDeviceToHost);

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, A.rows(), B.cols());

  // free memory
  cublasDestroy(handle);

  auto fun_free = [&pinned](T *x) { (pinned) ? cudaFreeHost(x) : cudaFree(x); };

  fun_free(dA);
  fun_free(dB);
  fun_free(dC);

  return C;
}

template <typename T> void benchmark(Mat<T> A, Mat<T> B, bool pinned = false) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  Mat<T> C = cublas_gemm(A, B);

  // outputs
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "Run time: " << elapsed_time.count() << " secs\n";
}

int main(int argc, char *argv[]) {

  // parse the input
  cxxopts::Options options(argv[0], "gemm example using eigen");
  options.positional_help("[optional args]").show_positional_help();
  options.add_options()("size", "dimension of the matrix",
                        cxxopts::value<int>()->default_value("100"));

  auto result = options.parse(argc, argv);
  int size = result["size"].as<int>();

  // Create CPU matrices
  Mat<float> A = Mat<float>::Random(size, size);
  Mat<float> B = Mat<float>::Random(size, size);

  std::cout << "Pageable Data Transfer\n";
  benchmark<float>(A, B);
  std::cout << "Pinned Data Transfer\n";
  benchmark<float>(A, B, true);

  return 0;
}
