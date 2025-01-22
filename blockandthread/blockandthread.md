# CUDA 线程和块结构详解

## 1. 基本概念

### 线程 (Thread)
最基本的执行单元，每个线程执行相同的代码但处理不同的数据。

### 块 (Block)
线程的集合，同一块内的线程可以：
- 共享内存
- 同步执行
- 相互通信

### 网格 (Grid)
块的集合，构成整个并行计算结构。

## 2. 线程层次结构

### 2.1 一维结构
```
Block 0:        Block 1:
[T0][T1][T2][T3] [T0][T1][T2][T3]
 0  1  2  3      4  5  6  7    <- 全局索引
```

```cpp
// 一维索引计算
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

### 2.2 二维结构
```
Block[0,0]             Block[1,0]
[T0,0][T1,0][T2,0]    [T0,0][T1,0][T2,0]
[T0,1][T1,1][T2,1]    [T0,1][T1,1][T2,1]

Block[0,1]             Block[1,1]
[T0,0][T1,0][T2,0]    [T0,0][T1,0][T2,0]
[T0,1][T1,1][T2,1]    [T0,1][T1,1][T2,1]
```

```cpp
// 二维索引计算
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int linear_idx = row * width + col;
```

## 3. 代码示例

### 3.1 一维配置
```cpp
// 启动配置：2个块，每个块4个线程
__global__ void kernel_1d() {
    int tid = threadIdx.x;    // 线程ID (0-3)
    int bid = blockIdx.x;     // 块ID (0-1)
    int gid = bid * blockDim.x + tid;  // 全局ID (0-7)
}

int main() {
    kernel_1d<<<2, 4>>>();  // 2块 × 4线程 = 8个总线程
}
```

### 3.2 二维配置
```cpp
// 启动配置：2×2块，每个块3×2线程
__global__ void kernel_2d() {
    // 块内位置
    int tx = threadIdx.x;  // 列索引
    int ty = threadIdx.y;  // 行索引
    
    // 块的位置
    int bx = blockIdx.x;   // 块列索引
    int by = blockIdx.y;   // 块行索引
}

int main() {
    dim3 blocks(2, 2);    // 2×2块网格
    dim3 threads(3, 2);   // 每块3×2线程
    kernel_2d<<<blocks, threads>>>();
}
```

## 4. 常见使用场景

### 4.1 一维结构
- 向量运算
- 一维数组处理
- 序列处理

### 4.2 二维结构
- 矩阵运算
- 图像处理
- 2D网格计算

## 5. 重要注意事项

1. **线程限制**：
   - 每个块的线程数有上限（通常是1024）
   - 不同GPU架构可能有不同限制

2. **性能考虑**：
   - 线程数最好是warp大小（32）的倍数
   - 块大小影响共享内存和同步开销

3. **内存访问**：
   - 同一warp的线程最好访问连续的内存
   - 合并访问可以提高内存带宽利用率