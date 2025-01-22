#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void blockandthread(){
    int one_dim = threadIdx.x;

    int row = threadIdx.y;
    int col = threadIdx.x;

    int two_dim = row * blockDim.x +col;
    printf("1D Position[%d]",one_dim);
    printf("2D Position[%d,%d] -> 1D Index: %d\n",row,col,two_dim);

}

__global__ void multi_block_thread() {
    // 获取块和线程索引
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    printf("Block[%d,%d], Thread[%d,%d]\n", 
           block_x, block_y, thread_x, thread_y);
}


int main(){
    dim3 blocks(2, 2);    // 2x2的块
    dim3 threads(3, 2);   // 每个块3x2的线程
    multi_block_thread<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
