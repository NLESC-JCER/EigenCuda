#include "eigencuda.hpp"
#include <chrono>
#include <cxxopts.hpp>
#include <iostream>

using eigencuda::Mat;

template <typename T> void benchmark(Mat<T> A, Mat<T> B, bool pinned = false) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  Mat<T> C = eigencuda::cublas_gemm(A, B, pinned);

  // outputs
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "Run time: " << elapsed_time.count() << " secs\n";
}

Mat<float> triple_product(Mat<float> A, Mat<float> B, Mat<float> C,
                    bool pinned = false) {
  // Perform the triple matrix Multiplication: A^T * B * C

  // Transfer the matrix matrix multiplacation of Eigen to GPU, using
  // CUBLas

  // Scalar constanst for calling blas
  constexpr float alpha = 1.;
  constexpr float beta = 0.;
  const float *pa = &alpha;
  const float *pb = &beta;

  // Size of the Matrices
  std::size_t size_A = A.rows() * A.cols() * sizeof(float);
  std::size_t size_B = B.rows() * B.cols() * sizeof(float);
  std::size_t size_C = C.rows() * C.cols() * sizeof(float);
  std::size_t size_X = B.rows() * C.cols() * sizeof(float);
  std::size_t size_Y = A.cols() * C.cols() * sizeof(float);

  Mat<float> X = Mat<float>::Zero(B.rows(), C.cols());
  Mat<float> Y = Mat<float>::Zero(A.cols(), C.cols());

  // and their pointers
  float *hA = A.data();
  float *hB = B.data();
  float *hC = C.data();
  float *hY = C.data();




  // alloc memory on the GPU
  float *dA, *dB, *dC, *dX, *dY;

  // Allocate either pageable or pinned memory
  auto fun_alloc = [&pinned](float **x, std::size_t n) {
    (pinned) ? cudaMallocHost(x, n) : cudaMalloc(x, n);
  };

  fun_alloc(&dA, size_A);
  fun_alloc(&dB, size_B);
  fun_alloc(&dC, size_C);
  fun_alloc(&dX, size_X);
  fun_alloc(&dY, size_Y);

  // cuda handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Transfer data to GPU
  cudaMemcpy(dA, hA, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, size_B, cudaMemcpyHostToDevice);
  cudaMemcpy(dC, hC, size_C, cudaMemcpyHostToDevice);

  // multiplied in the GPU
  // X = B * C
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.rows(), C.cols(), B.cols(),
              pa, dB, B.rows(), dC, C.rows(), pb, dX, X.rows());
  // R = A^T * X
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A.rows(), X.cols(), A.cols(),
              pa, dA, A.rows(), dX, X.rows(), pb, dY, Y.rows());

  // send data back to CPU
  cudaMemcpy(hY, dY, size_Y, cudaMemcpyDeviceToHost);

  // create an eigen matrix
  Y = Eigen::Map<Mat<float>>(hY, A.cols(), C.cols());

  // free memory
  cublasDestroy(handle);

  auto fun_free = [&pinned](float *x) {
    (pinned) ? cudaFreeHost(x) : cudaFree(x);
  };

  fun_free(dA);
  fun_free(dB);
  fun_free(dC);
  fun_free(dX);
  fun_free(dY);

  return Y;
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
  Mat<float> A = Mat<float>::Random(size + 10, size);
  Mat<float> B = Mat<float>::Random(size, size + 20);

  std::cout << "size: " << size << "\n";
  std::cout << "Pageable Data Transfer\n";
  benchmark<float>(A, B, false);
  std::cout << "Pinned Data Transfer\n";
  benchmark<float>(A, B, true);

  return 0;
}
