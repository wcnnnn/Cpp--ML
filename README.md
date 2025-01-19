# 机器学习算法 C++/CUDA实现

本项目旨在使用C++和CUDA实现经典机器学习算法和GPU并行计算，包括基础的机器学习模型、深度学习模型以及CUDA并行编程示例。通过C++和CUDA的实现深入理解算法原理和并行计算概念。

## 项目结构

```
.
├── README.md
├── blockandthread.cu     # CUDA线程块示例
├── utils/               # 工具类
│   ├── MatrixOps.h      # 矩阵运算接口
│   ├── MatrixOps.cpp    # CPU矩阵运算实现
│   ├── MatrixOpsCUDA.cu # GPU矩阵运算实现
│   ├── MatrixOpsCUDA.cuh # GPU矩阵运算头文件
│   └── matrix_benchmark.md # 性能对比文档
├── NeuralNetwork/       # 神经网络
│   ├── NeuralNetwork.cpp # CPU版本实现
│   ├── NeuralNetwork.cu  # GPU加速版本（进行中）
│   └── NeuralNetwork.md  # 实现文档
├── static/
│   ├── Perceptron/      # 感知机
│   │   ├── Perceptron.cpp
│   │   └── Perceptron.md
│   ├── KNN/             # k近邻法
│   │   ├── KNN.cpp
│   │   └── KNN.md
│   ├── NaiveBayes/      # 朴素贝叶斯（进行中）
│   ├── CNN/             # 卷积神经网络（计划中）
│   ├── RNN/             # 循环神经网络（计划中）
│   ├── DecisionTree/    # 决策树（计划中）
│   └── SVM/             # 支持向量机（计划中）
```

## CUDA并行计算示例

### 1. 线程块示例 (blockandthread.cu)
- 实现特点：
  - 演示CUDA的线程和块的组织结构
  - 包含1D和2D线程索引计算
  - 多块多线程的并行处理示例

### 2. GPU加速矩阵运算 (utils/MatrixOpsCUDA.cu)
- 状态：已完成基础实现
- 实现特点：
  - 五种基本矩阵运算的GPU加速实现
    1. 矩阵乘矩阵 (Matrix-Matrix Multiplication)
    2. 矩阵乘向量 (Matrix-Vector Multiplication)
    3. 向量乘矩阵 (Vector-Matrix Multiplication)
    4. 向量点乘 (Vector Dot Product)
    5. 向量外积 (Vector Outer Product)
  - 使用CUDA线程块和网格进行并行计算
  - 针对不同运算采用不同的并行策略
  - 支持大规模矩阵运算
  - 实现了性能测试基准

- 性能优化：
  1. 内存访问优化
     - 使用一维数组存储，采用行主序
     - 合理设计线程块大小
  2. 并行计算优化
     - 矩阵运算使用2D线程块
     - 向量运算使用1D线程块
  3. 规约优化
     - 向量点乘使用共享内存
     - 使用原子操作确保结果正确性

- 性能对比：
  1. 矩阵乘法性能
     | 矩阵规模 | CPU 时间(ms) | GPU 时间(ms) | 加速比 |
     |----------|--------------|--------------|--------|
     | 100x100  | 4           | 167          | 0.024x |
     | 500x500  | 604         | 10           | 60.4x  |
     | 1000x1000| 5365        | 53           | 101.2x |

  2. 向量运算性能
     - 大规模运算（>500维）可获得显著加速
     - 小规模运算（<500维）CPU性能更优
     - 保持了高精度（误差<1e-9）

### 3. 神经网络GPU加速 (NeuralNetwork/NeuralNetwork.cu)
- 状态：开发中
- 计划特点：
  - 前向传播的GPU加速
  - 反向传播的并行计算
  - 批量训练的GPU优化
  - 与CPU版本的性能对比

## 已实现算法

### 1. 感知机 (Perceptron)
- 实现文件：`static/Perceptron/Perceptron.cpp`
- 文档说明：`static/Perceptron/Perceptron.md`
- 实现特点：
  - 原始形式感知机算法
  - 使用C++ STL容器
  - 支持可配置的学习率和最大迭代次数
  - 包含AND门的实例演示

### 2. K近邻法 (KNN)
- 实现文件：`static/KNN/KNN.cpp`
- 文档说明：`static/KNN/KNN.md`
- 实现特点：
  - 使用KD树优化近邻搜索
  - 支持多种距离度量方法（欧氏距离、曼哈顿距离、切比雪夫距离）
  - 实现数据标准化（Z-score和Min-Max方法）
  - 包含分类任务的完整示例
  - 支持批量预测和准确率计算

### 3. 神经网络 (Neural Network)
- CPU实现：`NeuralNetwork/NeuralNetwork.cpp`
- GPU实现：`NeuralNetwork/NeuralNetwork.cu`（进行中）
- 文档说明：`NeuralNetwork/NeuralNetwork.md`
- 实现特点：
  - 支持多层前馈神经网络
  - 实现多种激活函数（Sigmoid、ReLU、Tanh、Softmax）
  - 支持批量训练
  - 包含矩阵运算工具类
  - 实现了XOR问题的完整示例
  - 支持Dropout正则化
  - GPU加速支持（开发中）

### 4. 朴素贝叶斯 (Naive Bayes)
- 状态：正在实现中
- 计划特点：
  - 支持离散和连续特征
  - 实现多种概率分布模型
  - 支持拉普拉斯平滑
  - 文本分类示例

## 开发环境

- 编程语言：C++ 11/14/17, CUDA 12.4
- 编译器：
  - C++: MinGW-w64
  - CUDA: NVCC + MSVC
- IDE：Visual Studio Code
- 构建系统：CMake（计划中）

## 使用说明

### 编译运行
```bash
# 编译CUDA示例
nvcc blockandthread.cu -o blockandthread
./blockandthread

# 编译GPU加速的矩阵运算
nvcc utils/MatrixOpsCUDA.cu -o matrix_ops_gpu
./matrix_ops_gpu

# 编译GPU加速的神经网络（开发中）
nvcc NeuralNetwork/NeuralNetwork.cu utils/MatrixOpsCUDA.cu -o nn_gpu
./nn_gpu

# 编译感知机示例
g++ static/Perceptron/Perceptron.cpp -o perceptron
./perceptron

# 编译KNN示例
g++ static/KNN/KNN.cpp -o knn
./knn

# 编译CPU版本神经网络
g++ NeuralNetwork/NeuralNetwork.cpp utils/MatrixOps.cpp -o nn_cpu
./nn_cpu
```

## 待实现功能

1. CUDA并行计算
   - [x] 线程块基础示例
   - [x] 矩阵运算GPU加速（进行中）
   - [x] 神经网络GPU加速（进行中）
   - [ ] 卷积运算GPU加速
   - [ ] 批量数据处理优化

2. 基础机器学习算法
   - [x] 感知机
   - [x] K近邻法
   - [ ] 朴素贝叶斯（进行中）
   - [ ] 决策树
   - [ ] 支持向量机（SVM）
   - [ ] 随机森林
   - [ ] GBDT
   - [ ] XGBoost

3. 深度学习算法
   - [x] 神经网络
   - [ ] CNN（计划中）
   - [ ] RNN（计划中）
   - [ ] LSTM
   - [ ] Transformer

4. 工程改进
   - [ ] 添加CMake构建系统
   - [ ] 添加单元测试
   - [ ] 添加性能测试
   - [ ] 支持数据集加载
   - [ ] 添加交叉验证
   - [ ] 可视化支持
   - [x] GPU加速支持（进行中）

## 性能对比

### 1. CPU vs GPU 矩阵运算性能
- 矩阵乘法：
  - 小规模（<500x500）：CPU更具优势
  - 大规模（>500x500）：GPU可获得50-100倍加速
  - 最优规模：1000x1000时达到最佳加速比

- 向量运算：
  - 点乘：规模>1M时GPU显示优势
  - 外积：规模>500x500时GPU更优
  - 数据传输开销是关键因素

### 2. 优化建议
- 小规模运算（<500维）建议使用CPU实现
- 大规模运算（>500维）建议使用GPU实现
- 批量运算时优先考虑GPU实现
- 需要考虑数据传输开销

## 贡献指南

欢迎贡献代码或提出建议！如果你想贡献代码，请：

1. Fork 本仓库
2. 创建新的分支
3. 提交你的修改
4. 创建 Pull Request

## 许可证

MIT License

## 参考资料

1. 《机器学习》- 周志华
2. 《深度学习》- Ian Goodfellow
3. C++ 参考手册
4. C++ STL 文档
5. CUDA Programming Guide
6. CUDA Best Practices Guide 