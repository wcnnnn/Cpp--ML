#include "../include/Optimizer.h"

void SGD::update(
    std::vector<std::vector<double>>& weights,
    std::vector<double>& bias,
    const std::vector<std::vector<double>>& weightGradients,
    const std::vector<double>& biasGradients
) {
    // 更新权重
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] -= learningRate * (weightGradients[i][j] + weightDecay * weights[i][j]);
        }
    }
    
    // 更新偏置
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learningRate * biasGradients[i];
    }
}

void SGD::update3D(
    std::vector<std::vector<std::vector<double>>>& weights,
    std::vector<double>& bias,
    const std::vector<std::vector<std::vector<double>>>& weightGradients,
    const std::vector<double>& biasGradients
) {
    // 更新权重
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            for (size_t k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] -= learningRate * (weightGradients[i][j][k] + weightDecay * weights[i][j][k]);
            }
        }
    }
    
    // 更新偏置
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] -= learningRate * biasGradients[i];
    }
}