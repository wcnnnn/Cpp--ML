# 矩阵运算性能对比分析

## 实现概述

本项目实现了基础矩阵运算的 CPU 和 GPU 版本，包括：
1. 矩阵乘矩阵 (Matrix-Matrix Multiplication)
2. 矩阵乘向量 (Matrix-Vector Multiplication)
3. 向量乘矩阵 (Vector-Matrix Multiplication)
4. 向量点乘 (Vector Dot Product)
5. 向量外积 (Vector Outer Product)

## 实现方法对比

| 运算类型 | CPU 实现 (C++) | GPU 实现 (CUDA) |
|----------|---------------|----------------|
| 矩阵乘矩阵<br>(C = A×B) | • 使用三重循环遍历<br>• 数据结构：`vector<vector<double>>`<br>• 核心代码：<br>`for(i = 0; i < m; i++)`<br>&nbsp;&nbsp;`for(j = 0; j < n; j++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`for(k = 0; k < p; k++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`C[i][j] += A[i][k]*B[k][j]`<br>• 时间复杂度：O(mnp) | • 使用二维线程块和网格<br>• 数据结构：一维数组，行主序存储<br>• 每个线程计算一个输出元素<br>• 核心代码：<br>`row = blockIdx.y * blockDim.y + threadIdx.y`<br>`col = blockIdx.x * blockDim.x + threadIdx.x`<br>`if(row < m && col < n)`<br>&nbsp;&nbsp;`for(k = 0; k < p; k++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`sum += A[row*p+k] * B[k*n+col]`<br>• 线程配置：`dim3 blockSize(16, 16)`<br>• 并行度：O(mn) |
| 矩阵乘向量<br>(y = Ax) | • 使用二重循环遍历<br>• 数据结构：矩阵用二维vector，向量用一维<br>• 核心代码：<br>`for(i = 0; i < m; i++)`<br>&nbsp;&nbsp;`for(j = 0; j < n; j++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`y[i] += A[i][j]*x[j]`<br>• 时间复杂度：O(mn) | • 使用一维线程块<br>• 每个线程处理一行的计算<br>• 核心代码：<br>`row = blockIdx.x * blockDim.x + threadIdx.x`<br>`if(row < m)`<br>&nbsp;&nbsp;`for(j = 0; j < n; j++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`sum += A[row*n+j] * x[j]`<br>• 线程配置：`blockSize = 256`<br>• 并行度：O(m) |
| 向量乘矩阵<br>(y = xA) | • 使用二重循环遍历<br>• 按列计算，避免缓存失效<br>• 核心代码：<br>`for(j = 0; j < n; j++)`<br>&nbsp;&nbsp;`for(i = 0; i < m; i++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`y[j] += x[i]*A[i][j]`<br>• 时间复杂度：O(mn) | • 使用一维线程块<br>• 每个线程处理一列的计算<br>• 核心代码：<br>`col = blockIdx.x * blockDim.x + threadIdx.x`<br>`if(col < n)`<br>&nbsp;&nbsp;`for(i = 0; i < m; i++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`sum += x[i] * A[i*n+col]`<br>• 线程配置：`blockSize = 256`<br>• 并行度：O(n) |
| 向量点乘<br>(c = x·y) | • 使用单重循环<br>• 直接累加计算<br>• 核心代码：<br>`double sum = 0.0;`<br>`for(i = 0; i < n; i++)`<br>&nbsp;&nbsp;`sum += x[i]*y[i]`<br>• 时间复杂度：O(n) | • 使用并行规约算法<br>• 共享内存优化<br>• 核心代码：<br>`__shared__ double temp[256]`<br>`temp[tid] = (i < n) ? x[i]*y[i] : 0`<br>`for(stride = blockDim.x/2; stride > 0; stride >>= 1)`<br>&nbsp;&nbsp;`if(tid < stride)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`temp[tid] += temp[tid + stride]`<br>• 使用 atomicAdd 合并结果<br>• 并行度：O(n) |
| 向量外积<br>(C = xy^T) | • 使用二重循环<br>• 直接计算每个元素<br>• 核心代码：<br>`for(i = 0; i < m; i++)`<br>&nbsp;&nbsp;`for(j = 0; j < n; j++)`<br>&nbsp;&nbsp;&nbsp;&nbsp;`C[i][j] = x[i]*y[j]`<br>• 时间复杂度：O(mn) | • 使用二维线程块<br>• 每个线程计算一个输出元素<br>• 核心代码：<br>`row = blockIdx.y * blockDim.y + threadIdx.y`<br>`col = blockIdx.x * blockDim.x + threadIdx.x`<br>`if(row < m && col < n)`<br>&nbsp;&nbsp;`C[row*n+col] = x[row]*y[col]`<br>• 线程配置：`dim3 blockSize(16, 16)`<br>• 并行度：O(mn) |

## 性能对比

### 矩阵乘法
| 矩阵规模 | CPU 时间(ms) | GPU 时间(ms) | 加速比 |
|----------|--------------|--------------|--------|
| 100x100  | 4           | 167          | 0.024x |
| 500x500  | 604         | 10           | 60.4x  |
| 1000x1000| 5365        | 53           | 101.2x |

### 向量点乘
| 向量规模 | CPU 时间(ms) | GPU 时间(ms) | 加速比 | 误差      |
|----------|--------------|--------------|--------|-----------|
| 1000     | <1          | <1           | -      | 1.42e-13  |
| 100000   | <1          | <1           | -      | 2.18e-11  |
| 1000000  | 2           | 2            | 1.0x   | 3.43e-9   |

## 关键优化技术

### GPU 优化策略
1. **内存访问优化**
   - 使用一维数组代替二维数组，减少内存访问开销
   - 合理设计线程块大小，提高内存访问效率

2. **并行计算优化**
   - 矩阵乘法：使用二维线程块和网格
   - 向量运算：使用一维线程块，适合向量操作

3. **规约优化**
   - 向量点乘使用共享内存进行并行规约
   - 使用 atomicAdd 确保结果正确性

## 性能分析

1. **规模效应**
   - 小规模运算：CPU 更快（GPU 数据传输开销大）
   - 大规模运算：GPU 显著优势（并行计算优势体现）

2. **加速效果**
   - 矩阵乘法：规模增大，加速比显著提升
   - 向量运算：规模较小时加速效果不明显

3. **精度分析**
   - 所有运算保持高精度
   - 向量点乘误差随规模增大而增加，但仍在可接受范围


