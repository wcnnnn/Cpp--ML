#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>
#include <iostream>

namespace MatrixOps {

// 矩阵乘法
std::vector<std::vector<double>> matrix_matrix_multiply(
    const std::vector<std::vector<double>>& x,
    const std::vector<std::vector<double>>& y);

// 矩阵向量乘法
std::vector<double> matrix_vector_multiply(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>& y);

// 向量矩阵乘法
std::vector<double> vector_matrix_multiply(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>& y);

// 向量点积
double vector_dot_product(
    const std::vector<double>& x,
    const std::vector<double>& y);

// 向量外积
std::vector<std::vector<double>> vector_outer_product(
    const std::vector<double>& x,
    const std::vector<double>& y);

} // namespace MatrixOps

#endif // MATRIX_OPS_H 