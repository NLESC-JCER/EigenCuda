# EigenCuda

This library is **Work in Progress**. There is no documentation!

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

  
  * [CXXOPTS](https://anaconda.org/conda-forge/cxxopts)
  * [Cudatoolkit](https://anaconda.org/anaconda/cudatoolkit)
  * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Usage

```
./cublas --size 100
```

where `size` is the size of the matrix 

## Example of performance

![alt text](./perf_gemm_gpu.png)
