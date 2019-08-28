#include "eigencuda.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using eigencuda::Mat;

void write_vector(std::vector<int> sizes,
                  std::vector<std::tuple<double, double>> vs) {
  std::ofstream file;
  double x, y;

  file.open("times_benchmark.txt");
  for (unsigned i = 0; i < vs.size(); i++) {
    std::tie(x, y) = vs[i];
    file << sizes[i] << " " << x << " " << y << "\n";
  }
  file.close();
}

template <typename T>
std::tuple<double, double> benchmark_right_matrix_tensor(
    Mat<T> &A, std::vector<Mat<T>> tensor) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double> elapsed_time;
 
  // Cuda
  start = std::chrono::system_clock::now();
  eigencuda::EigenCuda<double> EC;
  std::vector<Mat<double>> rs = EC.right_matrix_tensor(A, tensor);
  end = std::chrono::system_clock::now();
  elapsed_time = end - start;
  auto gpu_time = elapsed_time.count();
  std::cout << "GPU tensor matrix product: " << gpu_time << " secs\n";  

  // Call Eigen
  start = std::chrono::system_clock::now();
  std::vector<Mat<double>> qs;
  for (const auto &x : tensor) {
    Mat<double> r = x * A;
    qs.push_back(r);
  }
  end = std::chrono::system_clock::now();
  elapsed_time = end - start;
  auto cpu_time = elapsed_time.count();
  std::cout << "CPU tensor tensor product: " << cpu_time << " secs\n";
  
  return std::make_tuple(gpu_time, cpu_time);
}

void run_benchmark(std::vector<int> vs, bool pinned = false) {

  std::vector<std::tuple<double, double>> times;

  for (auto size : vs) {
    // Create CPU matrices
    std::cout << "running benchmark!\n";

    Mat<double> B = Mat<double>::Random(size, size + 20);

    std::cout << "size: " << size << "\n";

    // Benchmark for tensor product
    std::vector<Mat<double>> tensor;
    for (auto i = 0; i < 10; i++) {
      tensor.push_back(Mat<double>::Random(size + 20, size));
    }
    std::cout << "Running right matrix tensor benchmark\n";
    times.push_back(benchmark_right_matrix_tensor(B, tensor));
  }
  write_vector(vs, times);
}

void right_matrix_tensor() {
  // Define matrices and class to handle GPU resources
  eigencuda::EigenCuda<double> EC;

  // Call matrix multiplication GPU
  Mat<double> A = Mat<double>::Zero(2, 2);
  Mat<double> B = Mat<double>::Zero(3, 2);
  Mat<double> C = Mat<double>::Zero(3, 2);
  Mat<double> D = Mat<double>::Zero(3, 2);

  // Define matrices
  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8., 9., 10.;
  C << 9., 10., 11., 12., 13., 14.;
  D << 13., 14., 15., 16., 17., 18.;

  std::vector<Mat<double>> tensor{B, C, D};
  std::vector<Mat<double>> rs = EC.right_matrix_tensor(A, tensor);

  for (auto &x : rs) {
    std::cout << "result sum: " << x << "\n";
  }

  std::cout << "right matrix product succeeded!\n";
}

int main() {

  bool pinned = false;
  std::vector<int> vs{100, 200, 500};

  run_benchmark(vs, pinned);
  // right_matrix_tensor();
  return 0;
}
