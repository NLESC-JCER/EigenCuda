# EigenCuda

There is no CMAKE not test nor documentation and the code is ugly ! 

## Installation 

Use the Makefile in each directory

## Dependencies

  * Eigen
  * CCXXOPTS : https://anaconda.org/conda-forge/cxxopts

## Usage

```
./cublas --size 100
```

where `size` is the size of the matrix 

## Example of performance

![alt text](./perf_gemm_gpu.png)
