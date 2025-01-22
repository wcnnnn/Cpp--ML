#include "MatrixOps.h"

namespace MatrixOps {

std::vector<std::vector<double>> matrix_matrix_multiply(
    const std::vector<std::vector<double>>& x,
    const std::vector<std::vector<double>>& y) {
    if(x[0].size() != y.size()) {
        std::cout << "matrix size error" << std::endl;
        return std::vector<std::vector<double>>();
    }
    std::vector<std::vector<double>> z(x.size(), std::vector<double>(y[0].size()));
    for (size_t i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < y[0].size(); j++) {
            for (size_t k = 0; k < y.size(); k++) {
                z[i][j] += x[i][k] * y[k][j];
            }
        }
    }
    return z;
}

std::vector<double> matrix_vector_multiply(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>& y) {
    if (x[0].size() != y.size()) {
        std::cout << "matrix_vector size error" << std::endl;
        return std::vector<double>();
    }
    std::vector<double> z(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        for (size_t k = 0; k < y.size(); k++) {
            z[i] += x[i][k] * y[k];
        }
    }
    return z;
}

std::vector<double> vector_matrix_multiply(
    const std::vector<std::vector<double>>& x,
    const std::vector<double>& y) {
    if (x.size() != y.size()) {
        std::cout << "vector_matrix size error" << std::endl;
        return std::vector<double>();
    }
    std::vector<double> z(x[0].size());
    for (size_t j = 0; j < x[0].size(); j++) {
        for (size_t i = 0; i < y.size(); i++) {
            z[j] += x[i][j] * y[i];
        }
    }
    return z;
}

double vector_dot_product(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    if(x.size() != y.size()) {
        std::cout << "vector size error" << std::endl;
        return 0.0;
    }
    double z = 0;
    for (size_t i = 0; i < x.size(); i++) {
        z += x[i] * y[i];
    }
    return z;
}

std::vector<std::vector<double>> vector_outer_product(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    std::vector<std::vector<double>> z(x.size(), std::vector<double>(y.size()));
    for (size_t i = 0; i < x.size(); i++) {
        for (size_t j = 0; j < y.size(); j++) {
            z[i][j] = x[i] * y[j];
        }
    }
    return z;
}

} // namespace MatrixOps 