#include "eigencuda.hpp"
#include <chrono>
#include <cxxopts.hpp>
#include <iostream>

using eigencuda::Mat;

template <typename T>
void benchmark(Mat<T> A, Mat<T> B, Mat<T> C, bool pinned = false) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  // Mat<T> R = eigencuda::cublas_gemm(A, B, C, pinned);
  Mat<T> R = eigencuda::triple_product(A, B, C, pinned);

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
  Mat<float> A = Mat<float>::Random(size, size + 10);
  Mat<float> B = Mat<float>::Random(size, size + 20);
  Mat<float> C = Mat<float>::Random(size + 20, size);

  std::cout << "size: " << size << "\n";
  std::cout << "Pageable Data Transfer\n";
  // benchmark<float>(A, B, false);
  benchmark<float>(A, B, C, false);
  std::cout << "Pinned Data Transfer\n";
  // benchmark<float>(A, B, true);
  benchmark<float>(A, B, C, true);

  return 0;
}
