# 机器学习算法 C++/CUDA实现

本项目旨在使用C++和CUDA实现经典机器学习算法和GPU并行计算，包括基础的机器学习模型、深度学习模型以及CUDA并行编程示例。通过C++和CUDA的实现深入理解算法原理和并行计算概念。

## 项目结构

| 目录/文件 | 说明 | 状态 |
|-----------|------|------|
| [README.md](README.md) | 项目说明文档 | ✅ 已完成 |
| [blockandthread/](blockandthread/) | CUDA线程块示例 | ✅ 已完成 |
| **utils/** | **工具类库** | |
| &nbsp;&nbsp;&nbsp;&nbsp;[Common/](utils/Common/) | 通用工具 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[include/Activation.h](utils/Common/include/Activation.h) | 激活函数（ReLU、Sigmoid、Tanh、Softmax） | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[include/Loss.h](utils/Common/include/Loss.h) | 损失函数（MSE、交叉熵） | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[include/Optimizer.h](utils/Common/include/Optimizer.h) | 优化器（SGD） | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[Layers/](utils/Layers/) | 网络层实现 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[include/Layer1D.h](utils/Layers/include/Layer1D.h) | 1D层定义（全连接层、Dropout层） | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[include/Layer3D.h](utils/Layers/include/Layer3D.h) | 3D卷积层定义 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[CNNOps/](utils/CNNOps/) | CNN操作 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[MatrixOps/](utils/MatrixOps/) | CPU矩阵运算 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[MatrixOpsCUDA/](utils/MatrixOpsCUDA/) | GPU矩阵运算 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[ImageOps/](utils/ImageOps/) | 图像处理 | 🚧 计划中 |
| **CNN/** | **卷积神经网络实现** | |
| &nbsp;&nbsp;&nbsp;&nbsp;[CNN.cpp](CNN/CNN.cpp) | CNN主实现 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[CNN.md](CNN/CNN.md) | CNN实现说明文档 | ✅ 已完成 |
| **NeuralNetwork/** | **全连接神经网络** | |
| &nbsp;&nbsp;&nbsp;&nbsp;[NeuralNetwork.cpp](NeuralNetwork/NeuralNetwork.cpp) | CPU版本实现 | ✅ 已完成 |
| &nbsp;&nbsp;&nbsp;&nbsp;[NeuralNetwork.cu](NeuralNetwork/NeuralNetwork.cu) | GPU版本实现 | 🚧 进行中 |
| **Perception/** | **感知机** | |
| &nbsp;&nbsp;&nbsp;&nbsp;[Perceptron.cpp](Perception/Perceptron.cpp) | 感知机算法实现 | ✅ 已完成 |
| **KNN/** | **k近邻法** | |
| &nbsp;&nbsp;&nbsp;&nbsp;[KNN.cpp](KNN/KNN.cpp) | KNN算法实现 | ✅ 已完成 |
| **NaiveBayes/** | **朴素贝叶斯** | |
| &nbsp;&nbsp;&nbsp;&nbsp;[NaiveBayes.cpp](NaiveBayes/NaiveBayes.cpp) | 朴素贝叶斯实现 | 🚧 进行中 |
| **DecisionTree/** | **决策树** | 📋 计划中 |
| **SVM/** | **支持向量机** | 📋 计划中 |
| **RNN/** | **循环神经网络** | 📋 计划中 |


## 最新更新

### 1. CNN模块更新
- 实现3D卷积层，支持多通道输入输出
- 添加批量训练支持
- 实现交叉熵损失函数
- 优化内存管理和性能
- 完整的二分类示例实现

## 开发环境

- 编程语言：C++ 11/14/17, CUDA 12.4
- 编译器：
  - C++: MinGW-w64
  - CUDA: NVCC + MSVC
- IDE：Visual Studio Code

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
