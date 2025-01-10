# KNN算法实现与KD树优化

## C++知识点

### 1. 类与结构体
- 类的定义与成员访问控制（public/private）
  ```cpp
  class KNN {
  private:
      int k;
      int dimension;
  public:
      KNN(int initial_k, int initial_dimension, ...);
  };
  ```
- 构造函数与参数验证
  - 使用初始化列表
  - 输入参数合法性检查
- 结构体KDTree的实现
  - 数据成员和指针成员
  - 递归构造函数

### 2. STL容器的使用
- `vector<vector<double>>`: 二维动态数组存储训练数据
- `vector<int>`: 存储标签
- `map<int, int>`: 用于统计最近邻中各类别的频次
- `pair<double,int>`: 存储距离和对应的标签
- 容器的基本操作：
  - push_back()
  - resize()
  - size()
  - sort()

### 3. 算法与数据处理
- `algorithm`头文件的使用
  - sort(): 排序
  - min()/max(): 取最值
- `cmath`头文件的数学函数
  - pow(): 计算幂
  - sqrt(): 计算平方根
  - fabs(): 绝对值

### 4. 错误处理与输入验证
- 参数合法性检查
- 输出错误信息
- 边界条件处理

## 算法实现细节

### 1. KNN核心实现
- 构造函数
  ```cpp
  KNN(int initial_k, int initial_dimension, 
      vector<vector<double>> initial_train_data,
      vector<int> initial_target_label)
  ```
- 预测函数
  - 单样本预测
  - 批量预测
- 准确率计算

### 2. KD树优化
- 数据结构设计
  ```cpp
  struct KDTree {
      vector<double> data;
      int label;
      int split_dimension;
      KDTree* left;
      KDTree* right;
  };
  ```
- 构建过程
  1. 选择分割维度
  2. 按维度排序
  3. 选择中位数
  4. 递归构建左右子树
- 搜索过程
  1. 递归搜索最近邻
  2. 回溯检查
  3. 维护K个最近邻

### 3. 数据预处理功能
- 标准化方法
  1. Min-Max标准化
     ```cpp
     (x - min) / (max - min)
     ```
  2. Z-score标准化
     ```cpp
     (x - mean) / std
     ```
- 距离计算方法
  1. 欧氏距离
  2. 曼哈顿距离
  3. 切比雪夫距离

### 4. 性能优化
- KD树加速近邻搜索
- 批量预测
- 数据预处理
- 参数验证

## 使用示例

```cpp
// 创建分类器
KNN knn(3, 2, train_data, target_label);

// 数据标准化
vector<vector<double>> standardized_data = 
    knn.standardize(train_data, "zscore");

// 预测
vector<int> predictions = knn.predict_batch(test_data);

// 计算准确率
double accuracy = knn.accuracy(test_data, test_label);
```

## 可能的改进方向
1. 添加交叉验证
2. 实现K值自动选择
3. 支持特征权重
4. 添加并行计算支持
5. 实现数据集分割功能