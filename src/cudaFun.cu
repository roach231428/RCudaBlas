#include <cmath>
#include "cudaFun.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>


__global__ void mat_mul(const double* mat1, const double* mat2, double* result, const int M, const int N, const int S) {
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (threadId < M * S)
    {
        int row = threadId / M;
        int column = threadId % M;
        
        result[threadId] = 0;
        for (int j = 0; j < N; j++)
        {
            // result[threadId] += mat1[row * N + j] * mat2[j * S + column];
            result[threadId] += mat1[j * M + column] * mat2[row * N + j];
        }
    }
}


void matrix_multiplication_gpu(const double* mat1, const double* mat2, double* result, const int M, const int N, const int S)
{
    double *g_mat1, *g_mat2, *g_mat_result;
    
    cudaMalloc((void **)&g_mat1, sizeof(double) * M*N);
    cudaMalloc((void **)&g_mat2, sizeof(double) * N*S);
    cudaMalloc((void **)&g_mat_result, sizeof(double) * M*S);

    cudaMemcpy(g_mat1, mat1, sizeof(double) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(double) * N*S, cudaMemcpyHostToDevice);
    
    dim3 blockSize(32, 32);
    dim3 gridSize(20, 20);
    
    mat_mul<<< gridSize, blockSize >>>(g_mat1, g_mat2, g_mat_result, M, N, S);

    cudaMemcpy(result, g_mat_result, sizeof(double) * M*S, cudaMemcpyDeviceToHost);
    
    cudaFree(g_mat1);
    cudaFree(g_mat2);
    cudaFree(g_mat_result);
}