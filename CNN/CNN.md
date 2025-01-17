# 卷积神经网络（CNN）实现

## 算法原理

### 1. 基本概念

#### 1.1 网络结构
- 卷积层（Convolutional Layer）
- 池化层（Pooling Layer）
- 全连接层（Fully Connected Layer）

#### 1.2 核心组件

1. 卷积层
   - 卷积核（Kernel/Filter）
   - 步长（Stride）
   - 填充（Padding）
   - 特征图（Feature Map）

2. 池化层
   - 最大池化（Max Pooling）
   - 平均池化（Average Pooling）
   - 池化窗口大小
   - 步长

3. 激活函数
   - ReLU: max(0, x)
   - Sigmoid: 1/(1 + e^(-x))
   - Tanh: tanh(x)

### 2. 工作流程

#### 2.1 前向传播
1. 卷积操作
   - 卷积核与输入数据的点积
   - 生成特征图
   - 应用激活函数

2. 池化操作
   - 降采样
   - 保留重要特征
   - 减少计算量

3. 全连接层
   - 特征展平
   - 线性变换
   - 分类/回归

#### 2.2 反向传播
1. 计算损失梯度
2. 更新全连接层参数
3. 反向传播到池化层
4. 更新卷积核参数

## 代码实现

### 1. 类的设计

```cpp
// 基础层类
class Layer {
protected:
    LayerType type;
    ActivationType activation;
    vector<vector<vector<vector<double>>>> output;  // 4D张量
    vector<vector<vector<vector<double>>>> delta;   // 误差项
    
public:
    virtual void forward(const vector<vector<vector<vector<double>>>>& input) = 0;
    virtual void backward(const vector<vector<vector<vector<double>>>>& prev_delta) = 0;
};

// CNN类
class CNN {
private:
    vector<Layer*> layers;
    double learning_rate;
    
public:
    void add_conv_layer(const ConvParams& params, ActivationType activation);
    void add_pool_layer(const PoolParams& params);
    void add_fc_layer(int output_size, ActivationType activation);
    void train(const vector<vector<vector<vector<double>>>>& data,
              const vector<vector<double>>& targets,
              int epochs,
              int batch_size);
};
```

### 2. 核心功能

#### 2.1 卷积操作
- 实现四维张量的卷积运算
- 处理步长和填充
- 生成特征图

#### 2.2 池化操作
- 实现最大池化和平均池化
- 记录最大值位置（用于反向传播）
- 处理步长

#### 2.3 反向传播
- 实现链式法则
- 计算各层梯度
- 更新参数

## 使用示例

```cpp
// 创建CNN实例
CNN cnn(0.01);  // 学习率为0.01

// 添加网络层
ConvParams conv_params1 = {3, 1, 1, 16};  // 3x3卷积核，16个过滤器
cnn.add_conv_layer(conv_params1, ActivationType::RELU);

PoolParams pool_params = {2, 2, PoolType::MAX};  // 2x2最大池化
cnn.add_pool_layer(pool_params);

cnn.add_fc_layer(10, ActivationType::SIGMOID);  // 10分类问题

// 训练模型
cnn.train(training_data, targets, 100, 32);  // 100轮，批大小32

// 预测
auto result = cnn.predict(test_image);
```

## 优化方向

1. 性能优化
   - 使用CUDA加速
   - 实现矩阵运算优化
   - 内存管理优化

2. 功能扩展
   - 添加批归一化
   - 实现残差连接
   - 支持更多层类型

3. 训练优化
   - 实现动态学习率
   - 添加正则化
   - 实现早停机制

4. 工程改进
   - 支持模型保存/加载
   - 添加进度显示
   - 实现可视化工具 