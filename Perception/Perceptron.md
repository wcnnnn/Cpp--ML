# 感知机（Perceptron）实现文档

## C++知识点

### 1. 类的定义与实现
- 使用 `class` 关键字定义类
- 访问修饰符：`private` 和 `public`
- 构造函数的实现与参数默认值
- 成员变量和成员函数的声明与定义

### 2. STL容器使用
- `vector<double>` 用于存储权重和输入数据
- `vector<vector<double>>` 用于存储训练数据集
- 使用 `size()` 方法获取容器大小
- 使用下标操作符 `[]` 访问元素

### 3. 引用和常量
- `const vector<double>&` 使用常量引用作为参数，避免数据拷贝
- `const` 修饰符确保输入数据不被修改

### 4. 构造函数特性
- 参数默认值设置
- 成员初始化列表
- 向量初始化：`vector<double>(size, initial_value)`

## 感知机算法实现流程

### 1. 初始化
- 权重向量初始化为0.5
- 设置阈值（threshold）
- 设置偏置（bias）
- 设置学习率（learning_rate）
- 设置最大训练轮数（max_epoch）

### 2. 预测函数 `predict`
```cpp
if(inputs.size() != weights.size()) return -2;  // 输入检验
double weight_sum = bias;
for(size_t i = 0; i < inputs.size(); i++) {
    weight_sum += inputs[i] * weights[i];
}
return (weight_sum < threshold) ? -1 : 1;  // 阈值判断
```

### 3. 训练函数 `train`
1. 输入验证
   - 检查输入数据和目标值数量是否匹配
2. 训练循环
   - 外层循环：控制训练轮数（epoch）
   - 内层循环：遍历所有训练样本
3. 权重更新
   - 当预测结果错误时更新权重
   - 权重更新公式：`weight = weight + learning_rate * (target - result) * input`
   - 偏置更新公式：`bias = bias + learning_rate * (target - result)`
4. 收敛判断
   - 当所有样本预测正确时停止训练
   - 或达到最大训练轮数时停止

## 示例应用：AND门实现

### 训练数据
```cpp
vector<vector<double>> X = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};
vector<int> y = {-1, -1, -1, 1};  // AND门的期望输出
```

### 训练结果
- 经过少量迭代即可收敛
- 最终权重和偏置能够正确分类所有输入组合
- 测试结果显示完美的AND门行为

