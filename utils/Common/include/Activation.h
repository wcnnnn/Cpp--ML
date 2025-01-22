#pragma once
#include <vector>
#include <string>

enum class ActivationFunction {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

namespace Activation {
    // 前向传播的激活函数
    std::vector<double> forward(const std::vector<double>& input, 
                              const ActivationFunction& func_type);
    
    // 反向传播的激活函数导数
    std::vector<double> backward(const std::vector<double>& input, 
                               const ActivationFunction& func_type);
    
    // 辅助函数：将枚举转换为字符串
    std::string toString(const ActivationFunction& func_type);
}