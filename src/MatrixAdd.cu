#include <cmath>
#include "cudaFun.h"
#include "cuda_runtime.h"
#include <cuda.h>
#include <stdio.h>


__global__ void mat_add(const double* mat1, const double* mat2, double* result, const int M, const int N) {
    int iRow = blockDim.x * blockIdx.x + threadIdx.x;
    int iCol = blockDim.y * blockIdx.y + threadIdx.y;
    int threadId = iRow + iCol * N;
    
    if (threadId < M * N)
        result[threadId] = mat1[threadId] + mat2[threadId];
}


void matrix_add_gpu(const double* mat1, const double* mat2, double* result, const int M, const int N)
{
    double *g_mat1, *g_mat2, *g_mat_result;
    
    cudaMalloc((void **)&g_mat1, sizeof(double) * M*N);
    cudaMalloc((void **)&g_mat2, sizeof(double) * M*N);
    cudaMalloc((void **)&g_mat_result, sizeof(double) * M*N);

    cudaMemcpy(g_mat1, mat1, sizeof(double) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(double) * M*N, cudaMemcpyHostToDevice);
    
    dim3 blockSize(32, 32);
    dim3 gridSize(20, 20);
    
    mat_add<<< gridSize, blockSize >>>(g_mat1, g_mat2, g_mat_result, M, N);

    cudaMemcpy(result, g_mat_result, sizeof(double) * M*N, cudaMemcpyDeviceToHost);
    
    cudaFree(g_mat1);
    cudaFree(g_mat2);
    cudaFree(g_mat_result);
}