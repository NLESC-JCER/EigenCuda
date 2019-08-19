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

template <typename T> void dot_benchmark(Mat<T> &A, Mat<T> &B, bool pinned) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  eigencuda::EigenCuda<T> EC{pinned};
  Mat<T> R = EC.dot(A, B);

  // outputs
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "Run time: " << elapsed_time.count() << " secs\n";
}

template <typename T>
std::tuple<double, double>
benchmark_right_matrix_tensor(Mat<T> &A, std::vector<Mat<T>> tensor) {
  // chrono
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  eigencuda::EigenCuda<double> EC;
  std::vector<Mat<double>> rs = EC.right_matrix_tensor(A, tensor);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  auto gpu_time = elapsed_time.count();
  std::cout << "GPU triple tensor product: " << gpu_time << " secs\n";

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
  std::cout << "CPU triple tensor product: " << cpu_time << " secs\n";

  return std::make_tuple(gpu_time, cpu_time);
}

void run_benchmark(std::vector<int> vs, bool pinned = false) {

  std::vector<std::tuple<double, double>> times;

  for (auto size : vs) {
    // Create CPU matrices
    std::cout << "running benchmark!\n";

    Mat<double> A = Mat<double>::Random(size, size);
    Mat<double> B = Mat<double>::Random(size, size + 20);
    Mat<double> C = Mat<double>::Random(size + 20, size);

    std::string msg =
        (pinned) ? "Pinned Data Transfer" : "Pageable Data Transfer";
    std::cout << "size: " << size << "\n";
    std::cout << msg << "\n";

    dot_benchmark<double>(A, B, pinned);

    // Benchmark for triple product
    std::vector<Mat<double>> tensor;
    for (auto i = 0; i < 10; i++) {
      tensor.push_back(Mat<double>::Random(size, size + 20));
    }
    std::cout << "Running right matrix tensor benchmark\n";
    times.push_back(benchmark_right_matrix_tensor(C, tensor));
  }
  write_vector(vs, times);
}

void dot_product() {
  eigencuda::EigenCuda<double> EC;
  Mat<double> A = Mat<double>::Zero(2, 2);
  Mat<double> B = Mat<double>::Zero(2, 2);

  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8.;

  Mat<double> C = EC.dot(A, B);
  assert(abs(C.sum() - 134.) < 1e-8);
  std::cout << "dot product succeeded!\n";
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
  A << 1., 2., 3., 4., 5., 6.;
  B << 5., 6., 7., 8., 9., 10.;
  C << 9., 10., 11., 12., 13., 14., 15., 16., 17.;
  D << 13., 14., 15., 16., 17., 18., 19., 20., 21.;

  std::vector<Mat<double>> tensor{C, D};
  std::vector<Mat<double>> rs = EC.triple_tensor_product(A, B, tensor);

  // Check results
  assert(abs(rs[0].sum() - 12993.) < 1e-8);
  assert(abs(rs[1].sum() - 16773.) < 1e-8);

  std::cout << "triple product succeeded!\n";
}

void right_matrix_tensor() {
  // Define matrices and class to handle GPU resources
  eigencuda::EigenCuda<double> EC;

  // Call matrix multiplication GPU
  Mat<double> A = Mat<double>::Zero(2, 3);
  Mat<double> B = Mat<double>::Zero(3, 2);
  Mat<double> C = Mat<double>::Zero(3, 2);
  Mat<double> D = Mat<double>::Zero(3, 2);

  // Define matrices
  A << 1., 2., 3., 4., 5., 6.;
  B << 5., 6., 7., 8., 9., 10.;
  C << 9., 10., 11., 12., 13., 14.;
  D << 13., 14., 15., 16., 17., 18.;

  std::vector<Mat<double>> tensor{B};
  std::vector<Mat<double>> rs = EC.right_matrix_tensor(A, tensor);

  // Check results
  assert(abs(rs[0].sum() - 486.) < 1e-8);
  assert(abs(rs[1].sum() - 738.) < 1e-8);
  assert(abs(rs[1].sum() - 990.) < 1e-8);

  for (auto &x: rs){
    std::cout << "result sum: " << x << "\n";
  }
  
  std::cout << "right matrix product succeeded!\n";
}

void test() {
  Mat<double> A = Mat<double>::Ones(2, 2);
  Mat<double> B = Mat<double>::Zero(2, 2);

  B << 2, 3, 4, 5;

  Mat<double> rs = eigencuda::stack<double>(std::vector<Mat<double>>{A, B});
  std::cout << "result: " << rs << "\n";
}

int main() {

  bool pinned = false;
  std::vector<int> vs{100, 200, 500, 1000, 1500, 2000};

  // run_benchmark(vs, pinned);
  // dot_product();
  // triple_product();
  right_matrix_tensor();
  // test();
  return 0;
}
