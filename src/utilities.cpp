#include <Rcpp.h>
#include "cudaFun.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix matrix_multiplication_cuda(NumericMatrix mat1, NumericMatrix mat2){
  if(mat1.ncol() != mat2.nrow())
    stop("size of matrices mismatch.");
  int M = mat1.nrow();
  int N = mat1.ncol();
  int S = mat2.ncol();
  
  // std::vector<double> a = Rcpp::as< std::vector<double> >(mat1);
  // std::vector<double> b = Rcpp::as< std::vector<double> >(mat2);
  // std::vector<double> c(M*S, 0);
  // NumericMatrix returnMat(M, S);
  // double *c = &returnMat(0,0);
  
  // matrix_multiplication_gpu(&a[0], &b[0], &c[0], M, N, S);
  NumericMatrix returnMat(M, S);
  matrix_multiplication_gpu(&mat1(0,0), &mat2(0,0), &returnMat(0,0), M, N, S);
  
  // NumericVector returnMat = Rcpp::wrap(c);
  // returnMat.attr("dim") = Dimension(M, S);
  
  return returnMat;
}

// [[Rcpp::export]]
NumericMatrix matrix_multiplication_cuBLAs(NumericMatrix mat1, NumericMatrix mat2){
  if(mat1.ncol() != mat2.nrow())
    stop("size of matrices mismatch.");
  int M = mat1.nrow();
  int N = mat1.ncol();
  int S = mat2.ncol();

  double *a = &mat1(0,0);
  double *b = &mat2(0,0);
  NumericMatrix returnMat(M, S);
  double *c = &returnMat(0,0);

  double *g_mat1, *g_mat2, *g_mat_result;
  // cublasStatus_t stat;
  cublasHandle_t handle;

  cudaMalloc((void **)&g_mat1, sizeof(double) * M*N);
  cudaMalloc((void **)&g_mat2, sizeof(double) * N*S);
  cudaMalloc((void **)&g_mat_result, sizeof(double) * M*S);

  // initialize CUBLAS context
  cublasCreate(&handle);
  
  cublasSetMatrix(M, N, sizeof(*a), a, M, g_mat1, M);
  cublasSetMatrix(N, S, sizeof(*b), b, N, g_mat2, N);

  double al = 1.0;
  double bet = 0.0;

  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
              M, S, N, &al, g_mat1, M, g_mat2, N, &bet, 
              g_mat_result, M);
  cublasGetMatrix(M, S, sizeof(*c), g_mat_result, M, c, M);

  cudaFree(g_mat1);
  cudaFree(g_mat2);
  cudaFree(g_mat_result);
  
  return returnMat;
}