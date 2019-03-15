#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdlib.h>
#include <iostream>
#include <cublas_v2.h>
#include <chrono>
#include <curand.h>
#include <cxxopts.hpp>

//col Major for CUDA
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> Mat;

Mat cublas_gemm(Mat A, Mat B)
{

    const double alpha = 1.;
    const double beta = 0.;
    const double *pa = &alpha;
    const double *pb = &beta;

    int size = A.cols();
    Mat C = Mat::Zero(size,size);

    // and their pointers
    double *hA = A.data();
    double *hB = B.data();
    double *hC = C.data();

    // alloc memory on the GPU
    double *dA, *dB, *dC;
    cudaMalloc(&dA,size*size*sizeof(double));
    cudaMalloc(&dB,size*size*sizeof(double));
    cudaMalloc(&dC,size*size*sizeof(double));

    // cuda handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Transfer data to GPU
    cudaMemcpy(dA,hA,size*size*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,size*size*sizeof(double),cudaMemcpyHostToDevice);

    // process on GPU
    cublasDgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,size,size,size,pa,dA,size,dB,size,pb,dC,size);

    // send data back to CPU
    cudaMemcpy(hC,dC,size*size*sizeof(double),cudaMemcpyDeviceToHost);
    
    // create an eigen matrix
    C = Eigen::Map<Mat>(hC,size,size);

    // free memory
    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return C;
}


int main(int argc, char *argv[]) {
    
    // parse the input
    cxxopts::Options options(argv[0],  "gemm example using eigen");
    options.positional_help("[optional args]").show_positional_help();
    options.add_options()
        ("size", "dimension of the matrix", cxxopts::value<std::string>()->default_value("100"));

    auto result = options.parse(argc,argv);
    int size = std::stoi(result["size"].as<std::string>(),nullptr);

    // Create CPU matrices
    Mat A = Mat::Random(size,size);
    Mat B = Mat::Random(size,size);

    // chrono    
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    Mat C = cublas_gemm(A, B);

    // outputs
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = end-start;
    std::cout << "Run time    : " << elapsed_time.count() << " secs" <<  std::endl;

    return 0;
}
