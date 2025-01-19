#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <vector>
#include "MatrixOpsCUDA.cuh"

using namespace std;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

vector<double> Matrix2Array(const vector<vector<double>> & Matrix){
    int col = Matrix[0].size();
    int row = Matrix.size();
    vector<double> output(col*row);
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            output[i*col+j] = Matrix[i][j];
        }
    }
    return output;
}
vector<vector<double>> Array2Matrix(const vector<double>& Array,int row,int col){
    vector<vector<double>> output(row,vector<double>(col));
    for (size_t i = 0; i < row; i++)
    {
        for (size_t j = 0; j < col; j++)
        {
            output[i][j] = Array[i*col+j];
        }
    }
    return output;
}

namespace cuda {

__global__ void matrix_matrix_multiply(
double* A, double* B, double* C, int m, int n, int p){
    int row = blockIdx.y * blockDim.y+ threadIdx.y;
    int col = blockIdx.x * blockDim.x+ threadIdx.x;
    if (row<m && col<n)
    {
        double sum = 0.0;
        // 一维数组访问：row * 列数 + col
        for (size_t i = 0; i < p; i++)
        {
            sum+=A[row*p+i]*B[i*n+col];
        }
        C[row*n+col]=sum;
    }
};

__global__ void matrix_vector_multiply(
double*A,double*B,double*C,int m,int n){
    int row = blockIdx.x * blockDim.x+ threadIdx.x;
    if (row<m)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
        {
            sum+=A[row*n+i]*B[i];
        }
        C[row] = sum;
    }
};
__global__ void vector_matrix_multiply(
double*A,double*B,double*C,int m,int n){
    int col = blockIdx.x * blockDim.x+ threadIdx.x;
    if (col<n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < m; i++)
        {
            sum+=B[i]*A[i*n+col];
        }
        C[col] = sum;
    }
};

__global__ void vector_dot_product(
double* A, double* B, double* C, int m) {
    __shared__ double temp[256];  // 共享内存，用于规约
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程计算自己负责的部分
    temp[tid] = (i < m) ? A[i] * B[i] : 0.0;
    __syncthreads();  // 同步所有线程
    
    // 并行规约求和
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }
    // 只有第一个线程写入结果
    if (tid == 0) {
        atomicAdd(C, temp[0]);
    }
};

__global__ void vector_outer_product(
double*A,double*B,double*C,int m,int n){
    int row = blockIdx.y * blockDim.y+ threadIdx.y;
    int col = blockIdx.x * blockDim.x+ threadIdx.x;
    if (row<m && col<n)
    {
        C[row*n+col]=A[row]*B[col];
    }
};

} // namespace cuda
