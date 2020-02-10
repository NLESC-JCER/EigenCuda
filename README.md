[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3660936.svg)](https://doi.org/10.5281/zenodo.3660936)

# EigenCuda

Offload the [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) matrix matrix multiplication to an Nvidia GPU
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
#include "cudapipeline.hpp"

using eigencuda::CudaPipeline;
using eigencuda::CudaMatrix;

  // Call the class to handle GPU resources
  CudaPipeline cuda_pip;

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
std::vector<Eigen::MatrixXd> results(3, Eigen::MatrixXd::Zero(3, 2));
CudaMatrix cuma_A{A, cuda_pip.get_stream()};
CudaMatrix cuma_B{3, 2, cuda_pip.get_stream()};
CudaMatrix cuma_C{3, 2, cuda_pip.get_stream()};

for (Index i = 0; i < 3; i++) {
  cuma_B.copy_to_gpu(tensor[i]);
  cuda_pip.gemm(cuma_B, cuma_A, cuma_C);
  results[i] = cuma_C;
}
// Expected results
bool pred_1 = X.isApprox(results[0]);
bool pred_2 = Y.isApprox(results[1]);
bool pred_3 = Z.isApprox(results[2]);
}
```
