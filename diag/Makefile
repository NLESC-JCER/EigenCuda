

INCS_CXXOPTS = -I/home/nicolas/cxxopts/include
INCS_EIGEN = -I/home/nicolas/eigen4cuda/include/eigen3/

all: cusolver cusolver_double

cusolver: diag_cusolver.cpp
	nvcc ${INCS_CXXOPTS} ${INCS_EIGEN} $< -o $@ -lcusolver -lcublas -lcurand 

cusolver_double: diag_cusolver_double.cpp
	nvcc ${INCS_CXXOPTS} ${INCS_EIGEN} $< -o $@ -lcusolver -lcublas -lcurand 