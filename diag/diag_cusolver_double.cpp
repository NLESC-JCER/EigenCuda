#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <stdlib.h>
#include <iostream>

#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>

#include <chrono>
#include <cxxopts.hpp>

// https://docs.nvidia.com/cuda/cusolver/index.html#syevd-example1

//col Major for CUDA
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> Mat;


Mat cusolver_diag(Mat A)
{
    int size = A.cols();

    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;

    const int m = size;
    const int lda = m;    

    Mat V = Mat::Zero(size,size);
    Mat W = Mat::Zero(size,1);

    // and their pointers
    double *hA = A.data();
    double *hW = W.data();

    // some stuff
    double * d_A = NULL;
    double * d_W = NULL;

    int * devInfo = NULL;
    double *d_work = NULL;
    int lwork = 0;
    int info_gpu = 0;

    // step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    // step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&devInfo, sizeof(int));

    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // Transfer data to GPU
    cudaStat1 = cudaMemcpy(d_A, hA, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnDsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

    // step 4: compute spectrum
    cusolver_status = cusolverDnDsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // copy data to host
    cudaStat1 = cudaMemcpy(hW, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    //cudaStat2 = cudaMemcpy(hV, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    //assert(cudaSuccess == cudaStat2);

    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat3);
    
    // create an eigen matrix
    W = Eigen::Map<Mat>(hW,size,size);

    // free resources
    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);

    cudaDeviceReset();  

    return W;

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
    
    // chrono    
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    Mat B = cusolver_diag(A);

    // outputs
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = end-start;
    std::cout << "Run time    : " << elapsed_time.count() << " secs" <<  std::endl;

    return 0;
}