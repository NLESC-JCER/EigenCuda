#ifndef EIGENCUDA_H_
#define EIGENCUDA_H_

#include "cudatensorbase.hpp"
#include <curand.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
/*
 * \brief Perform Tensor-matrix multiplications in a GPU
 *
 * The `CudaPipeline` class handles the allocation and deallocation of arrays on
 * the GPU.
 */

namespace eigencuda {

class CudaMatrix : CudaTensorBase {
 public:
  Index size() const override { return _rows * _cols; };
  Index rows() const { return _rows; };
  Index cols() const { return _cols; };
  double *data() const { return _data.get(); };

  CudaMatrix(const Eigen::MatrixXd &matrix, const cudaStream_t &stream);

  // Allocate memory in the GPU for a matrix
  CudaMatrix(Index nrows, Index ncols, const cudaStream_t &stream);

  // Convert A Cudamatrix to an EigenMatrix
  operator Eigen::MatrixXd() const;

  void copy_to_gpu(const Eigen::MatrixXd &A);

 private:
  Index _rows;
  Index _cols;
};

}  // namespace eigencuda

#endif  // EIGENCUDA_H_
