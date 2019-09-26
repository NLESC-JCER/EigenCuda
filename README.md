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
