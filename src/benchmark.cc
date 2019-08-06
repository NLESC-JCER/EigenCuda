#include "eigencuda.hpp"
#include "matmul.hpp"
#include <chrono>
#include <cxxopts.hpp>
#include <iostream>
#include <vector>

using eigencuda::Mat;

template <typename T>
void dot_benchmark(Mat<T> &A, Mat<T> &B, Mat<T> &C, bool pinned) {
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

template <typename T>
void triple_product_benchmark(Mat<T> &A, Mat<T> &B,
                              std::vector<Mat<T>> tensor) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  eigencuda::EigenCuda<double> EC;
  std::vector<Mat<double>> rs = EC.triple_tensor_product(A, B, tensor);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "GPU triple tensor product: " << elapsed_time.count()
            << " secs\n";

  // Call Eigen
  start = std::chrono::system_clock::now();
  std::vector<Mat<double>> qs;
  for (const auto &x : tensor) {
    Mat<double> r = A * (x * B);
    qs.push_back(r);
  }
  end = std::chrono::system_clock::now();
  elapsed_time = end - start;
  std::cout << "CPU triple tensor product: " << elapsed_time.count()
            << " secs\n";
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
    dot_benchmark<double>(A, B, C, pinned);

    // Benchmark for triple product
    std::vector<Mat<double>> tensor;
    for (auto i = 0; i < 10; i++) {
      tensor.push_back(Mat<double>::Random(size, size + 20));
    }
    std::cout << "Running triple product benchmark\n";
    triple_product_benchmark(A, C, tensor);
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
  // Define matrices and class to handle GPU resources
  eigencuda::EigenCuda<double> EC;
  // Call matrix multiplication GPU
  Mat<double> A = Mat<double>::Zero(2, 3);
  Mat<double> B = Mat<double>::Zero(3, 2);
  Mat<double> C = Mat<double>::Zero(3, 3);
  Mat<double> D = Mat<double>::Zero(3, 3);

  // Define matrices
  // A << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
  // B << 5., 6., 7., 8., 9., 10., 11., 12., 13.;
  A << 1., 2., 3., 4., 5., 6.;
  B << 5., 6., 7., 8., 9., 10.;
  C << 9., 10., 11., 12., 13., 14., 15., 16., 17.;
  D << 13., 14., 15., 16., 17., 18., 19., 20., 21.;

  std::vector<Mat<double>> tensor{C, D};
  std::vector<Mat<double>> rs = EC.triple_tensor_product(A, B, tensor);

  for (const auto &x : rs) {
    std::cout << "vector: " << x << "\n";
    std::cout << "sum: " << x.sum() << "\n";
  }

  // auto size = 50;
  // Mat<double> A = Mat<double>::Random(size, size);
  // Mat<double> B = Mat<double>::Random(size+20, size);
  // Mat<double> C = Mat<double>::Random(size + 20, size);
  // Mat<double> D = Mat<double>::Random(size + 20, size);

  // triple_product_benchmark(A, B, tensor);

  // // Check results
  // // assert(abs(rs[0].sum() - 2854.) < 1e-8);
  // // assert(abs(rs[1].sum() - 3894.) < 1e-8);
  // std::cout << "triplet product seems to be correct\n";
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

  // run_benchmark(vs, pinned);
  // dot_product();
  triple_product();
  return 0;
}
