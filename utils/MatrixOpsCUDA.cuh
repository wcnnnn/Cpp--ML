#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

// 工具函数声明
std::vector<double> Matrix2Array(const std::vector<std::vector<double>>& Matrix);
std::vector<std::vector<double>> Array2Matrix(const std::vector<double>& Array, int row, int col);

namespace cuda {

// 矩阵乘矩阵
__global__ void matrix_matrix_multiply(
    double* A, double* B, double* C, 
    int m, int n, int p);

// 矩阵乘向量
__global__ void matrix_vector_multiply(
    double* A, double* B, double* C,
    int m, int n);

// 向量乘矩阵
__global__ void vector_matrix_multiply(
    double* A, double* B, double* C,
    int m, int n);

// 向量点乘
__global__ void vector_dot_product(
    double* A, double* B, double* C,
    int m);

// 向量外积
__global__ void vector_outer_product(
    double* A, double* B, double* C,
    int m, int n);

} // namespace cuda

#endif // MATRIX_OPS_CUH
