#pragma once
#include <vector>
#include <string>

enum class LossFunction {
    MSE,
    RMSE,
    CROSS_ENTROPY
};

namespace Loss {
    // 计算损失
    double compute(const std::vector<double>& output,
                  const std::vector<double>& target,
                  const LossFunction& loss_type);
    
    // 计算损失导数
    std::vector<double> derivative(const std::vector<double>& output,
                                 const std::vector<double>& target,
                                 const LossFunction& loss_type);
    
    // 辅助函数：将枚举转换为字符串
    std::string toString(const LossFunction& loss_type);
}