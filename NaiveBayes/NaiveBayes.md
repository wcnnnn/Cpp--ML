# 朴素贝叶斯算法实现

## 算法原理

### 1. 基本概念
朴素贝叶斯算法是基于贝叶斯定理与特征条件独立假设的分类方法。

#### 贝叶斯定理
P(Y|X) = P(X|Y)P(Y) / P(X)

其中：
- P(Y|X) 是后验概率：在特征X出现的情况下类别Y出现的概率
- P(X|Y) 是条件概率：在类别Y的条件下特征X出现的概率
- P(Y) 是先验概率：类别Y出现的概率
- P(X) 是证据因子：特征X出现的概率

#### 条件独立性假设
朴素贝叶斯的"朴素"体现在假设所有特征之间相互独立：
P(X|Y) = P(X₁|Y) × P(X₂|Y) × ... × P(Xₙ|Y)

### 2. 实现要点

#### 2.1 处理离散特征
- 计算每个类别的先验概率 P(Y)
- 计算每个特征值在每个类别下的条件概率 P(X|Y)
- 使用拉普拉斯平滑处理零概率问题

#### 2.2 处理连续特征
- 假设特征服从高斯分布
- 计算每个类别下每个特征的均值和方差
- 使用高斯分布概率密度函数计算条件概率

#### 2.3 预测过程
1. 计算每个类别的后验概率
2. 选择后验概率最大的类别作为预测结果
3. 使用对数概率避免数值下溢

## 代码实现

### 1. 类的设计
```cpp
class NaiveBayes {
private:
    map<int, double> prior_probability;  // 先验概率
    map<int, vector<pair<double, double>>> continuous_params;  // 连续特征参数
    map<int, map<int, map<double, double>>> discrete_params;  // 离散特征参数
    vector<bool> feature_types;  // 特征类型（连续/离散）
    
public:
    NaiveBayes(const vector<bool>& feature_types);
    void fit(const vector<vector<double>>& X, const vector<int>& y);
    int predict(const vector<double>& x);
    vector<int> predict_batch(const vector<vector<double>>& X);
    double accuracy(const vector<vector<double>>& X, const vector<int>& y);
};
```

### 2. 核心功能
1. 模型训练（fit）
   - 计算先验概率
   - 计算条件概率
   - 处理连续和离散特征

2. 预测功能（predict）
   - 计算后验概率
   - 选择最优类别

3. 工具函数
   - 高斯分布概率计算
   - 对数概率计算
   - 准确率评估

## 使用示例

```cpp
// 创建分类器（指定特征类型：true为连续，false为离散）
vector<bool> feature_types = {true, false, true};
NaiveBayes nb(feature_types);

// 训练模型
nb.fit(train_data, train_labels);

// 预测新样本
vector<double> new_sample = {1.5, 0, 2.1};
int prediction = nb.predict(new_sample);

// 批量预测
vector<int> predictions = nb.predict_batch(test_data);

// 计算准确率
double acc = nb.accuracy(test_data, test_labels);
```

## 优化方向

1. 特征选择
   - 信息增益
   - 互信息
   - 卡方检验

2. 数值计算优化
   - 使用对数概率避免数值溢出
   - 并行计算支持

3. 模型改进
   - 支持特征权重
   - 处理缺失值
   - 增加交叉验证

4. 工程优化
   - 增加数据预处理
   - 支持模型序列化
   - 添加进度显示 