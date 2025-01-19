#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>
#include "MatrixOps.h"
#include "MatrixOpsCUDA.cuh"

using namespace std;
using namespace std::chrono;

// 生成随机矩阵
vector<vector<double>> generate_random_matrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols));
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    return matrix;
}

// 生成随机向量
vector<double> generate_random_vector(int size) {
    vector<double> vec(size);
    for(int i = 0; i < size; i++) {
        vec[i] = (double)rand() / RAND_MAX;
    }
    return vec;
}

// 测试矩阵乘法性能
void test_matrix_multiplication(int m, int n, int p) {
    cout << "\nTesting matrix multiplication (" << m << "x" << n << ") * (" << n << "x" << p << ")" << endl;
    
    // 生成测试数据
    auto A = generate_random_matrix(m, n);
    auto B = generate_random_matrix(n, p);
    
    // CPU版本测试
    auto start = high_resolution_clock::now();
    vector<vector<double>> C_cpu = MatrixOps::matrix_matrix_multiply(A, B);
    auto end = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end - start).count();
    cout << "CPU time: " << cpu_time << " ms" << endl;
    
    // GPU版本测试
    start = high_resolution_clock::now();
    
    // 转换为一维数组
    auto A_array = Matrix2Array(A);
    auto B_array = Matrix2Array(B);
    vector<double> C_array(m * p);
    
    // 分配GPU内存
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_array.size() * sizeof(double));
    cudaMalloc(&d_B, B_array.size() * sizeof(double));
    cudaMalloc(&d_C, C_array.size() * sizeof(double));
    
    // 复制数据到GPU
    cudaMemcpy(d_A, A_array.data(), A_array.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_array.data(), B_array.size() * sizeof(double), cudaMemcpyHostToDevice);
    
    // 设置CUDA网格和块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((p + blockSize.x - 1) / blockSize.x, 
                  (m + blockSize.y - 1) / blockSize.y);
    
    // 执行GPU计算
    cuda::matrix_matrix_multiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, p, n);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(C_array.data(), d_C, C_array.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    // 转换回二维矩阵
    auto C_gpu = Array2Matrix(C_array, m, p);
    
    end = high_resolution_clock::now();
    auto gpu_time = duration_cast<milliseconds>(end - start).count();
    cout << "GPU time: " << gpu_time << " ms" << endl;
    cout << "Acceleration ratio: " << (float)cpu_time / gpu_time << "x" << endl;
    
    // 清理GPU内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 测试向量点乘性能
void test_vector_dot_product(int size) {
    cout << "\nTesting vector dot product (size: " << size << ")" << endl;
    
    // 生成测试数据
    auto x = generate_random_vector(size);
    auto y = generate_random_vector(size);
    
    // CPU版本测试
    auto start = high_resolution_clock::now();
    double result_cpu = MatrixOps::vector_dot_product(x, y);
    auto end = high_resolution_clock::now();
    auto cpu_time = duration_cast<milliseconds>(end - start).count();
    cout << "CPU time: " << cpu_time << " ms" << endl;
    
    // GPU版本测试
    start = high_resolution_clock::now();
    
    // 分配GPU内存
    double *d_x, *d_y, *d_result;
    cudaMalloc(&d_x, size * sizeof(double));
    cudaMalloc(&d_y, size * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));
    
    // 复制数据到GPU
    cudaMemcpy(d_x, x.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    
    // 设置CUDA参数
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // 执行GPU计算
    double result_gpu = 0;
    cudaMemcpy(d_result, &result_gpu, sizeof(double), cudaMemcpyHostToDevice);
    cuda::vector_dot_product<<<gridSize, blockSize>>>(d_x, d_y, d_result, size);
    cudaMemcpy(&result_gpu, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    
    end = high_resolution_clock::now();
    auto gpu_time = duration_cast<milliseconds>(end - start).count();
    cout << "GPU time: " << gpu_time << " ms" << endl;
    cout << "Acceleration ratio: " << (float)cpu_time / gpu_time << "x" << endl;
    
    // 验证结果
    cout << "Result error: " << abs(result_cpu - result_gpu) << endl;
    
    // 清理GPU内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
}

int main() {
    cout << "Starting matrix operations benchmark..." << endl;
    
    // 测试不同规模的矩阵乘法
    cout << "Testing matrix multiplication" << endl;
    test_matrix_multiplication(100, 100, 100);    // 小规模
    test_matrix_multiplication(500, 500, 500);    // 中规模
    test_matrix_multiplication(1000, 1000, 1000); // 大规模
    
    // 测试不同规模的向量点乘
    cout << "Testing vector dot product" << endl;
    test_vector_dot_product(1000);      // 小规模
    test_vector_dot_product(100000);    // 中规模
    test_vector_dot_product(1000000);   // 大规模
    
    cout << "Benchmark completed" << endl;
    return 0;
} 