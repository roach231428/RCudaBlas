# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

matrix_multiplication_cuda <- function(mat1, mat2) {
    .Call(`_RCudaBlas_matrix_multiplication_cuda`, mat1, mat2)
}

matrix_add_cuda <- function(mat1, mat2) {
    .Call(`_RCudaBlas_matrix_add_cuda`, mat1, mat2)
}

matrix_multiplication_cuBLAs <- function(mat1, mat2) {
    .Call(`_RCudaBlas_matrix_multiplication_cuBLAs`, mat1, mat2)
}

