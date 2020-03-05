
#include "memory_manager.hpp"
#include <cutensor.h>
#include <iostream>
#include <unordered_map>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

int main() {
  // Host element type definition
  using floatTypeCompute = float;

  // CUDA types
  cudaDataType_t typeA = CUDA_R_64F;
  cudaDataType_t typeB = CUDA_R_64F;
  cudaDataType_t typeC = CUDA_R_64F;
  cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_64F;

  floatTypeCompute alpha = 1.1;
  floatTypeCompute beta = 0.9;

  std::cout << "Include headers and define data types\n";

  // Create vector of modes
  std::vector<int> modeC{'m', 'u', 'n', 'v'};
  std::vector<int> modeA{'m', 'h', 'k', 'n'};
  std::vector<int> modeB{'u', 'k', 'v', 'h'};
  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();

  // Extents
  std::unordered_map<int, int64_t> extent;
  extent['m'] = 96;
  extent['n'] = 96;
  extent['u'] = 96;
  extent['v'] = 64;
  extent['h'] = 64;
  extent['k'] = 64;

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentC;
  for (auto mode : modeC) extentC.push_back(extent[mode]);
  std::vector<int64_t> extentA;
  for (auto mode : modeA) extentA.push_back(extent[mode]);
  std::vector<int64_t> extentB;
  for (auto mode : modeB) extentB.push_back(extent[mode]);

  // Number of elements of each tensor
  size_t elementsA = 1;
  for (auto mode : modeA) elementsA *= extent[mode];
  size_t elementsB = 1;
  for (auto mode : modeB) elementsB *= extent[mode];
  size_t elementsC = 1;
  for (auto mode : modeC) elementsC *= extent[mode];

  // Eigen Tensors
  Eigen::Tensor<double, 4> tensorA(96, 96, 64, 64);
  Eigen::Tensor<double, 4> tensorB(96, 64, 64, 64);
  Eigen::Tensor<double, 4> tensorC(96, 96, 96, 64);

  // Initialize tensor as Random
  tensorA.setRandom();
  tensorB.setRandom();
  tensorC.setRandom();

  // Size in bytes
  size_t sizeA = sizeof(double) * tensorA.size();
  size_t sizeB = sizeof(double) * tensorB.size();
  size_t sizeC = sizeof(double) * tensorC.size();

  // Allocate on device
  eigencuda::Unique_ptr_to_GPU_data A_in_gpu =
      eigencuda::alloc_tensor_in_gpu(sizeA);
  eigencuda::Unique_ptr_to_GPU_data B_in_gpu =
      eigencuda::alloc_tensor_in_gpu(sizeB);
  eigencuda::Unique_ptr_to_GPU_data C_in_gpu =
      eigencuda::alloc_tensor_in_gpu(sizeC);

  // // Initialize data on host
  // for (int64_t i = 0; i < elementsA; i++)
  //   A[i] = (((float)rand()) / RAND_MAX - 0.5) * 100;
  // for (int64_t i = 0; i < elementsB; i++)
  //   B[i] = (((float)rand()) / RAND_MAX - 0.5) * 100;
  // for (int64_t i = 0; i < elementsC; i++)
  //   C[i] = (((float)rand()) / RAND_MAX - 0.5) * 100;

  // Copy to device
  // cudaMemcpy(C_in_gpu, tensorC, sizeC, cudaMemcpyHostToDevice);
  // cudaMemcpy(A_in_gpu, tensorA, sizeA, cudaMemcpyHostToDevice);
  // cudaMemcpy(B_in_gpu, tensorB, sizeB, cudaMemcpyHostToDevice);
  eigencuda::checkCuda(cudaMemcpy(A_in_gpu.get(), tensorA.data(), sizeA,
                                  cudaMemcpyHostToDevice));
  eigencuda::checkCuda(cudaMemcpy(B_in_gpu.get(), tensorB.data(), sizeB,
                                  cudaMemcpyHostToDevice));

  eigencuda::checkCuda(cudaMemcpy(C_in_gpu.get(), tensorC.data(), sizeC,
                                  cudaMemcpyHostToDevice));

  std::cout << "Allocate, initialize and transfer tensors\n";

  return 0;
}