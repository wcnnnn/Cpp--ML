# 机器学习算法 C++实现

本项目旨在使用C++实现经典机器学习算法，包括基础的机器学习模型和深度学习模型。通过C++的实现，不仅可以深入理解算法原理，还能学习C++的高级特性和最佳实践。

## 项目结构

```
.
├── README.md
├── static/
│   ├── Perceptron/        # 感知机
│   │   ├── Perceptron.cpp
│   │   └── Perceptron.md
│   ├── KNN/               # k近邻法
│   │   ├── KNN.cpp
│   │   └── KNN.md
│   ├── NaiveBayes/        # 朴素贝叶斯（进行中）
│   ├── NeuralNetwork/     # 神经网络
│   │   ├── NeuralNetwork.cpp
│   │   └── NeuralNetwork.md
│   ├── CNN/               # 卷积神经网络（计划中）
│   ├── RNN/               # 循环神经网络（计划中）
│   ├── DecisionTree/      # 决策树（计划中）
│   ├── SVM/               # 支持向量机（计划中）
│   └── utils/             # 工具类
│       ├── MatrixOps.h    # 矩阵运算
│       └── MatrixOps.cpp
```

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
- 实现文件：`static/NeuralNetwork/NeuralNetwork.cpp`
- 文档说明：`static/NeuralNetwork/NeuralNetwork.md`
- 实现特点：
  - 支持多层前馈神经网络
  - 实现多种激活函数（Sigmoid、ReLU、Tanh、Softmax）
  - 支持批量训练
  - 包含矩阵运算工具类
  - 实现了XOR问题的完整示例
  - 支持Dropout正则化

### 4. 朴素贝叶斯 (Naive Bayes)
- 状态：正在实现中
- 计划特点：
  - 支持离散和连续特征
  - 实现多种概率分布模型
  - 支持拉普拉斯平滑
  - 文本分类示例

## 开发环境

- 编程语言：C++ 11/14/17
- 编译器：MinGW-w64
- IDE：Visual Studio Code
- 构建系统：CMake（计划中）

## 代码规范

1. 命名规范
   - 类名：大驼峰命名法（PascalCase）
   - 函数名：小驼峰命名法（camelCase）
   - 变量名：下划线命名法（snake_case）
   - 常量：全大写加下划线

2. 文档规范
   - 每个算法实现都配有详细的markdown文档
   - 文档包含算法原理、代码实现要点和使用示例
   - 关键函数都有详细的注释说明

3. 代码组织
   - 每个算法独立成类
   - 算法相关的工具函数放在相应的命名空间中
   - 测试用例和示例代码分开存放

## 使用说明

### 编译运行
```bash
# 编译感知机示例
g++ static/Perceptron/Perceptron.cpp -o perceptron
./perceptron

# 编译KNN示例
g++ static/KNN/KNN.cpp -o knn
./knn

# 编译神经网络示例（需要包含矩阵运算库）
g++ static/NeuralNetwork/NeuralNetwork.cpp static/utils/MatrixOps.cpp -o nn
./nn
```

## 待实现功能

1. 基础机器学习算法
   - [x] 感知机
   - [x] K近邻法
   - [ ] 朴素贝叶斯（进行中）
   - [ ] 决策树
   - [ ] 支持向量机（SVM）
   - [ ] 随机森林
   - [ ] GBDT
   - [ ] XGBoost

2. 深度学习算法
   - [x] 神经网络
   - [ ] CNN（计划中）
   - [ ] RNN（计划中）
   - [ ] LSTM
   - [ ] Transformer

3. 工程改进
   - [ ] 添加CMake构建系统
   - [ ] 添加单元测试
   - [ ] 添加性能测试
   - [ ] 支持数据集加载
   - [ ] 添加交叉验证
   - [ ] 可视化支持
   - [ ] GPU加速支持

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