#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <curand.h>
#include <cxxopts.hpp>
#include <iostream>
#include <stdlib.h>

// Check Cuda error
inline
cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}



// col Major for CUDA
template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
Mat<T> cublas_gemm(Mat<T> A, Mat<T> B, bool pinned=false) {

  constexpr T alpha = 1.;
  constexpr T beta = 0.;
  const T *pa = &alpha;
  const T *pb = &beta;

  int size = A.cols();
  Mat<T> C = Mat<T>::Zero(size, size);

  // and their pointers
  T *hA = A.data();
  T *hB = B.data();
  T *hC = C.data();

  // alloc memory on the GPU
  T *dA, *dB, *dC;
  if (pinned){
    cudaMallocHost(&dA, size * size * sizeof(T));
    cudaMallocHost(&dB, size * size * sizeof(T));
    cudaMallocHost(&dC, size * size * sizeof(T));
  } else {
    cudaMalloc(&dA, size * size * sizeof(T));
    cudaMalloc(&dB, size * size * sizeof(T));
    cudaMalloc(&dC, size * size * sizeof(T));
}

  // cuda handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Transfer data to GPU
  cudaMemcpy(dA, hA, size * size * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size * size * sizeof(T), cudaMemcpyHostToDevice);

  // process on GPU
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, pa, dA, size,
              dB, size, pb, dC, size);
  // gpu_blas_gemm(handle,dA,dB,dC,size);

  // send data back to CPU
  cudaMemcpy(hC, dC, size * size * sizeof(T), cudaMemcpyDeviceToHost);

  // create an eigen matrix
  C = Eigen::Map<Mat<T>>(hC, size, size);

  // free memory
  cublasDestroy(handle);
  if (pinned) {
    cudaFreeHost(dA);
    cudaFreeHost(dB);
    cudaFreeHost(dC);
  } else {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
  return C;
}

template <typename T>
void benchmark(Mat<T> A, Mat<T> B, bool pinned=false) {
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
                        cxxopts::value<std::string>()->default_value("100"));

  auto result = options.parse(argc, argv);
  int size = std::stoi(result["size"].as<std::string>(), nullptr);

  // Create CPU matrices
  Mat<float> A = Mat<float>::Random(size, size);
  Mat<float> B = Mat<float>::Random(size, size);

  std::cout << "Pageable Data Transfer\n";
  benchmark<float>(A, B);
  std::cout << "Pinned Data Transfer\n";
  benchmark<float>(A, B, true);

  return 0;
}
