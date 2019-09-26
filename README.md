# EigenCuda

Offload the [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) matrix matrix multiplacation to an Nvidia GPU
using [cublas](https://docs.nvidia.com/cuda/cublas/index.html).

## CMake Installation

To compile execute:
```
cmake -H. -Bbuild && cmake --build build
```

To Debug compile as:
```
cmake -H. -Bbuild  -DCMAKE_BUILD_TYPE=Debug && cmake --build build
```

## Dependencies

This packages assumes that you have installed the following packages:
  
  * [Cudatoolkit](https://anaconda.org/anaconda/cudatoolkit)
  * [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Usage
### Matrix Multiplication
```cpp
#include "eigencuda.hpp"

eigencuda::EigenCuda EC;
Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2, 2);

A << 1., 2., 3., 4.;
B << 5., 6., 7., 8.;

Eigen::MatrixXd C = EC.matrix_mult(A, B);
assert(abs(C.sum() - 134.) < 1e-8);
```

### Tensor Matrix Multiplication
```
#include "eigencuda.hpp"
  // Call the class to handle GPU resources
  eigencuda::EigenCuda EC;

  // Call matrix multiplication GPU
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2, 2);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(3, 2);
  Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(3, 2);

  // Define matrices
  A << 1., 2., 3., 4.;
  B << 5., 6., 7., 8., 9., 10.;
  C << 9., 10., 11., 12., 13., 14.;
  D << 13., 14., 15., 16., 17., 18.;
  X << 23., 34., 31., 46., 39., 58.;
  Y << 39., 58., 47., 70., 55., 82.;
  Z << 55., 82., 63., 94., 71., 106.;

  std::vector<Eigen::MatrixXd> tensor{B, C, D};
  EC.right_matrix_tensor_mult(std::move(tensor), A);

  // Expected results
  assert(X.isApprox(tensor[0]));
  assert(Y.isApprox(tensor[1]));
  assert(Z.isApprox(tensor[2]));
``