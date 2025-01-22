# 机器学习算法 C++/CUDA实现

本项目旨在使用C++和CUDA实现经典机器学习算法和GPU并行计算，包括基础的机器学习模型、深度学习模型以及CUDA并行编程示例。通过C++和CUDA的实现深入理解算法原理和并行计算概念。

## 项目结构

```
├── README.md                           # 项目说明文档
├── blockandthread.cu                   # CUDA线程块示例
├── utils/                             # 工具类库
│   ├── Common/                        # 通用工具
│   │   ├── include/                   # 头文件
│   │   │   ├── Activation.h          # 激活函数（ReLU、Sigmoid、Tanh、Softmax）
│   │   │   ├── Loss.h               # 损失函数（MSE、交叉熵）
│   │   │   └── Optimizer.h          # 优化器（SGD）
│   │   └── src/                      # 源文件
│   │       ├── Activation.cpp        # 激活函数实现
│   │       ├── Loss.cpp             # 损失函数实现
│   │       └── Optimizer.cpp        # 优化器实现
│   ├── Layers/                       # 网络层实现
│   │   ├── include/                  # 头文件
│   │   │   ├── Layer1D.h            # 1D层定义（全连接层、Dropout层）
│   │   │   └── Layer3D.h            # 3D卷积层定义（Conv3D、MaxPool3D、Linear3D）
│   │   └── src/                     # 源文件
│   │       ├── Layer1D.cpp          # 1D层实现
│   │       └── Layer3D.cpp          # 3D卷积层实现
│   ├── CNNOps/                      # CNN操作
│   │   ├── include/                 # 头文件
│   │   │   └── CNNOps.h            # CNN基础操作定义
│   │   └── src/                    # 源文件
│   │       └── CNNOps.cpp          # CNN基础操作实现（卷积、池化）
│   ├── MatrixOps/                  # CPU矩阵运算
│   │   ├── MatrixOps.h            # 矩阵运算接口定义
│   │   └── MatrixOps.cpp          # 矩阵运算CPU实现
│   ├── MatrixOpsCUDA/             # GPU矩阵运算
│   │   ├── MatrixOpsCUDA.cu      # 矩阵运算GPU实现
│   │   └── MatrixOpsCUDA.cuh     # GPU相关头文件
│   └── ImageOps/                  # 图像处理（计划中）
├── CNN/                          # 卷积神经网络实现
│   ├── CNN.cpp                  # CNN主实现（包含网络结构和训练逻辑）
│   └── CNN.md                   # CNN实现说明文档
├── NeuralNetwork/               # 全连接神经网络
│   ├── NeuralNetwork.cpp       # CPU版本实现
│   ├── NeuralNetwork.cu        # GPU版本实现（进行中）
│   └── NeuralNetwork.md        # 神经网络实现说明文档
├── Perceptron/                 # 感知机
│   ├── Perceptron.cpp         # 感知机算法实现
│   └── Perceptron.md          # 感知机说明文档
├── KNN/                       # k近邻法
│   ├── KNN.cpp               # KNN算法实现（包含KD树）
│   └── KNN.md                # KNN说明文档
├── NaiveBayes/               # 朴素贝叶斯（进行中）
│   ├── NaiveBayes.cpp       # 朴素贝叶斯实现
│   └── NaiveBayes.md        # 实现说明文档
├── DecisionTree/            # 决策树（计划中）
│   ├── DecisionTree.cpp    # 决策树实现
│   └── DecisionTree.md     # 实现说明文档
├── SVM/                    # 支持向量机（计划中）
│   ├── SVM.cpp            # SVM实现
│   └── SVM.md             # 实现说明文档
└── RNN/                   # 循环神经网络（计划中）
    ├── RNN.cpp           # RNN实现
    └── RNN.md            # 实现说明文档
```

## 最新更新

### 1. CNN模块更新
- 实现3D卷积层，支持多通道输入输出
- 添加批量训练支持
- 实现交叉熵损失函数
- 优化内存管理和性能
- 完整的二分类示例实现

### 2. 工具类改进
- 重构为模块化设计
- 分离通用组件（激活函数、损失函数、优化器）
- 实现3D卷积和池化操作
- 增强代码复用性

### 3. 性能优化
- 优化矩阵运算
- 改进梯度计算
- 提升内存效率
- 支持批量处理

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
  - 使用CUDA线程块和网格进行并行计算
  - 支持大规模矩阵运算
  - 实现了性能测试基准

## 已实现算法

### 1. 卷积神经网络 (CNN)
- 实现文件：`CNN/CNN.cpp`
- 文档说明：`CNN/CNN.md`
- 实现特点：
  - 支持3D卷积层（多通道输入输出）
  - 实现最大池化层
  - 支持全连接层
  - 实现多种激活函数（ReLU、Softmax等）
  - 支持交叉熵损失函数
  - 实现批量训练
  - 支持随机梯度下降优化器

### 2. 神经网络 (Neural Network)
- CPU实现：`NeuralNetwork/NeuralNetwork.cpp`
- GPU实现：`NeuralNetwork/NeuralNetwork.cu`（进行中）
- 实现特点：
  - 支持多层前馈神经网络
  - 实现多种激活函数（Sigmoid、ReLU、Tanh）
  - 支持批量训练
  - 实现了XOR问题的完整示例

### 3. 感知机 (Perceptron)
- 实现文件：`static/Perceptron/Perceptron.cpp`
- 实现特点：
  - 原始形式感知机算法
  - 支持可配置的学习率
  - 包含AND门的实例演示

### 4. K近邻法 (KNN)
- 实现文件：`static/KNN/KNN.cpp`
- 实现特点：
  - 使用KD树优化近邻搜索
  - 支持多种距离度量方法
  - 实现数据标准化

### 5. 朴素贝叶斯 (Naive Bayes)
- 状态：正在实现中
- 计划特点：
  - 支持离散和连续特征
  - 实现多种概率分布模型
  - 支持拉普拉斯平滑

## 工具类库

### 1. Common
- 激活函数：ReLU、Sigmoid、Tanh、Softmax
- 损失函数：MSE、交叉熵
- 优化器：随机梯度下降（SGD）

### 2. Layers
- Layer3D：支持3D卷积、池化和全连接操作
- 支持前向传播和反向传播
- 实现参数更新和梯度计算

### 3. CNNOps
- 实现卷积操作
- 实现池化操作
- 支持多通道处理

### 4. MatrixOps
- CPU和GPU矩阵运算实现
- 基础矩阵操作
- 优化的数值计算

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

# 编译CNN
g++ -std=c++11 CNN/CNN.cpp utils/Common/src/*.cpp utils/Layers/src/*.cpp utils/CNNOps/*.cpp utils/MatrixOps/*.cpp -I. -Iutils -o CNN/cnn.exe

# 编译神经网络
g++ -std=c++11 NeuralNetwork/NeuralNetwork.cpp utils/MatrixOps/*.cpp -o NeuralNetwork/neural_network.exe

# 编译感知机示例
g++ Perceptron/Perceptron.cpp -o perceptron

# 编译KNN示例
g++ KNN/KNN.cpp -o knn
```

## 待实现功能

1. CNN增强
   - [ ] 添加更多层类型
   - [ ] 支持模型保存和加载
   - [ ] 添加数据增强
   - [ ] GPU加速支持

2. 深度学习模型
   - [ ] RNN实现
   - [ ] LSTM实现
   - [ ] Transformer实现

3. 基础算法
   - [ ] 决策树
   - [ ] 支持向量机
   - [ ] 随机森林
   - [ ] GBDT

4. 工程改进
   - [ ] 添加CMake构建系统
   - [ ] 添加单元测试
   - [ ] 添加性能测试
   - [ ] 支持数据集加载
   - [ ] 添加交叉验证
   - [ ] 可视化支持

## 性能对比

### 1. CPU vs GPU 矩阵运算性能
- 矩阵乘法：
  - 小规模（<500x500）：CPU更具优势
  - 大规模（>500x500）：GPU可获得50-100倍加速
  - 最优规模：1000x1000时达到最佳加速比

### 2. 优化建议
- 小规模运算（<500维）建议使用CPU实现
- 大规模运算（>500维）建议使用GPU实现
- 批量运算时优先考虑GPU实现

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
3. 《神经网络与深度学习》- 邱锡鹏
4. C++ 参考手册
5. CUDA Programming Guide
6. CUDA Best Practices Guide 
