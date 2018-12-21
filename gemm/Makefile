

INCS_CXXOPTS = -I/home/nicolas/cxxopts/include
INCS_EIGEN = -I/home/nicolas/eigen4cuda/include/eigen3/

cublas: gemm_cublas.cpp
	nvcc ${INCS_CXXOPTS} ${INCS_EIGEN} $< -o $@ -lcublas -lcurand
