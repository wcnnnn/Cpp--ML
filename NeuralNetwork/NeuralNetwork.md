# 一维神经网络实现框架

## 一、项目结构
```
NeuralNetwork/
├── utils/
│   ├── MatrixOps.h      // 矩阵运算声明
│   └── MatrixOps.cpp    // 矩阵运算实现
└── NeuralNetwork/
    ├── NeuralNetwork.h  // 神经网络声明
    └── NeuralNetwork.cpp // 神经网络实现
```

## 二、基础工具实现

### 1. 矩阵运算（MatrixOps.h/cpp）
```cpp
namespace MatrixOps {
    // 1.1 矩阵乘法 (M x N) * (N x P) = (M x P)
    vector<vector<double>> matrix_matrix_multiply(
        const vector<vector<double>>& a,  // M x N
        const vector<vector<double>>& b   // N x P
    );

    // 1.2 矩阵向量乘法 (M x N) * (N) = (M)
    vector<double> matrix_vector_multiply(
        const vector<vector<double>>& matrix,  // M x N
        const vector<double>& vec              // N
    );

    // 1.3 向量矩阵乘法 (N) * (N x M) = (M)
    vector<double> vector_matrix_multiply(
        const vector<double>& vec,             // N
        const vector<vector<double>>& matrix   // N x M
    );

    // 1.4 向量点积
    double vector_dot_product(
        const vector<double>& a,
        const vector<double>& b
    );
}
```

### 2. 基础数据结构
```cpp
// 2.1 激活函数类型
enum class ActivationFunction {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

// 2.2 层类型
enum class Layers {
    Linear,
    Dropout
};

// 2.3 优化器类型
enum class Optimizer {
    SGD
};

// 2.4 损失函数类型
enum class Loss_function {
    MSE,
    RMSE
};
```

## 三、核心功能实现

### 1. 激活函数模块
```cpp
// 1.1 前向传播激活函数
vector<double> Activation(const vector<double>& x, const string& activate) {
    if (activate == "SIGMOID") {
        // sigmoid实现
    } else if (activate == "RELU") {
        // relu实现
    } else if (activate == "TANH") {
        // tanh实现
    } else if (activate == "SOFTMAX") {
        // softmax实现
    }
}

// 1.2 激活函数导数
vector<double> ActivationDerivative(const vector<double>& z, const string& activate) {
    if (activate == "SIGMOID") {
        // sigmoid导数
    } else if (activate == "RELU") {
        // relu导数
    } // ...
}
```

### 2. 损失函数模块
```cpp
// 2.1 损失计算
double Loss(const vector<double>& error, const string& loss_function) {
    if (loss_function == "MSE") {
        // MSE实现
    } else if (loss_function == "RMSE") {
        // RMSE实现
    }
}

// 2.2 损失函数导数
vector<double> LossDerivative(
    const vector<double>& output,
    const vector<double>& target,
    const string& loss_function
) {
    // 实现损失函数导数
}
```

### 3. 优化器模块
```cpp
class SGD {
private:
    double learning_rate;

public:
    SGD(double lr = 0.01);
    
    void update(
        vector<vector<double>>& weights,
        vector<double>& bias,
        const vector<vector<double>>& weight_gradients,
        const vector<double>& bias_gradients
    );
};
```

### 4. 网络层实现
```cpp
struct Layer {
    // 4.1 成员变量
    vector<vector<double>> weights;  // 权重矩阵
    vector<double> bias;            // 偏置向量
    vector<double> outputs;         // 输出值
    vector<double> deltas;          // 误差项
    vector<double> z;               // 激活前的值
    ActivationFunction activation;  // 激活函数类型
    Layers layer_type;             // 层类型
    
    // 4.2 构造函数
    Layer(const int input_size, const int output_size,
          ActivationFunction act_type, Layers layer_type);
    
    // 4.3 前向传播
    vector<double> make_layer(const vector<double>& input);
};
```

### 5. 神经网络类
```cpp
class NeuralNetwork {
private:
    // 5.1 成员变量
    vector<Layer> layers;           // 网络层
    vector<int> topology;          // 网络结构
    int batch_size;               // 批量大小
    double learning_rate;         // 学习率
    vector<double> inputs;        // 输入数据

public:
    // 5.2 构造函数
    NeuralNetwork(const vector<int>& topology,
                 int batch_size = 16,
                 double lr = 0.01);

    // 5.3 前向传播
    vector<double> forward(const vector<double>& input);

    // 5.4 反向传播
    vector<double> backward(const vector<double>& outputs,
                          const vector<double>& targets,
                          SGD& optimizer);

    // 5.5 训练函数
    void train(const vector<vector<double>>& data,
              const vector<vector<double>>& targets,
              int epochs,
              int batch_size);
};
```

## 四、实现顺序

1. 首先实现矩阵运算库
   - 实现基础矩阵运算函数
   - 添加维度检查
   - 测试矩阵运算正确性

2. 实现激活函数和损失函数
   - 实现各种激活函数及其导数
   - 实现损失函数及其导数
   - 测试函数的数值正确性

3. 实现优化器
   - 实现SGD优化器
   - 测试权重更新逻辑

4. 实现网络层
   - 实现Layer构造函数
   - 实现前向传播
   - 测试单层运算

5. 实现神经网络类
   - 实现网络构造
   - 实现前向传播
   - 实现反向传播
   - 实现训练函数

6. 测试验证
   - 使用XOR问题测试
   - 验证训练过程
   - 检查预测结果

## 五、关键点说明

1. 矩阵维度
   - 权重矩阵维度：(output_size x input_size)
   - 输入向量维度：(input_size)
   - 输出向量维度：(output_size)

2. 初始化方法
   - 使用He初始化：weights ~ N(0, sqrt(2/input_size))
   - 偏置初始化为0

3. 前向传播顺序
   - 矩阵乘法：z = weights * input + bias
   - 激活函数：output = activation(z)

4. 反向传播顺序
   - 计算输出层误差
   - 计算隐藏层误差
   - 计算权重梯度
   - 更新权重和偏置 