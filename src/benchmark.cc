#include "eigencuda.hpp"
#include <chrono>
#include <cxxopts.hpp>
#include <iostream>
#include <vector>

using eigencuda::Mat;

template <typename T>
void benchmark(Mat<T> A, Mat<T> B, Mat<T> C, bool pinned) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  eigencuda::EigenCuda<T> EC{pinned};
  // Mat<T> R = eigencuda::cublas_gemm(A, B, C, pinned);
  // Mat<T> R = eigencuda::triple_product(A, B, C, pinned);
  Mat<T> R = EC.dot(A, B);

  // outputs
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "Run time: " << elapsed_time.count() << " secs\n";
}

void run_benchmark(std::vector<int> vs, bool pinned = false) {
  for (auto size : vs) {
    // Create CPU matrices
    Mat<double> A = Mat<double>::Random(size, size);
    Mat<double> B = Mat<double>::Random(size, size + 20);
    Mat<double> C = Mat<double>::Random(size + 20, size);

    std::string msg =
        (pinned) ? "Pinned Data Transfer" : "Pageable Data Transfer";
    std::cout << "size: " << size << "\n";
    std::cout << msg << "\n";
    benchmark<double>(A, B, C, pinned);
  }
}

void dot_product() {
  eigencuda::EigenCuda<double> EC;
  Mat<double> A = Mat<double>::Zero(2, 2);
  Mat<double> B = Mat<double>::Zero(2, 2);

  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8.;

  Mat<double> C = EC.dot(A, B);
  std::cout << "dot product: " << C.sum() << "\n";
}

void triple_product() {
  eigencuda::EigenCuda<double> EC;
  Mat<double> A = Mat<double>::Zero(2, 2);
  Mat<double> B = Mat<double>::Zero(2, 2);
  Mat<double> C = Mat<double>::Zero(2, 2);
  Mat<double> D = Mat<double>::Zero(2, 2);

  // Define matrices
  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8.;
  C << 9., 10., 11., 12.;
  D << 13., 14., 15., 16.;

  std::vector<Mat<double>> tensor{C, D};
  std::vector<Mat<double>> rs = EC.triple_tensor_product(A, B, tensor);

  // Check results
  assert(abs(rs[0].sum() - 2854.) < 1e-8);
  assert(abs(rs[1].sum() - 3894.) < 1e-8);

  std::cout << "triple tensor product is done!" << "\n";
}

int main(int argc, char *argv[]) {

  // parse the input
  cxxopts::Options options(argv[0], "gemm example using eigen");
  options.positional_help("[optional args]").show_positional_help();
  options.add_options()("pinned", "Whether to use pinned memory",
                        cxxopts::value<bool>()->default_value("false"));

  auto result = options.parse(argc, argv);
  bool pinned = result["pinned"].as<bool>();

  std::vector<int> vs{100, 200, 500, 1000, 1500, 2000};

  run_benchmark(vs, pinned);
  dot_product();
  triple_product();
  return 0;
}
