#include "../include/Layer1D.h"
#include "../../Common/include/Activation.h"
#include "../../MatrixOps/MatrixOps.h"

Layer1D::Layer1D(const std::string& name, 
                 int input_size, 
                 int output_size,
                 Layer1DType type,
                 ActivationFunction act,
                 double dropout_rate)
    : Layer(name, act), 
      layer_type(type),
      dropout_rate(dropout_rate) {
    
    // 初始化权重和偏置
    weights.resize(output_size, std::vector<double>(input_size));
    bias.resize(output_size, 0.0);
    output.resize(output_size);
    z.resize(output_size);
    
    if (type == Layer1DType::Dropout) {
        dropout_mask.resize(output_size, true);
    }
    
    initialize();
}

void Layer1D::initialize() {
    // He初始化
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0/weights[0].size()));
    
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[0].size(); j++) {
            weights[i][j] = dist(gen);
        }
    }
}

std::vector<double> Layer1D::forward(const std::vector<double>& input) {
    this->input = input;  // 存储输入用于反向传播
    
    // 线性变换: z = Wx + b
    z = MatrixOps::matrix_vector_multiply(weights, input);
    for (size_t i = 0; i < z.size(); i++) {
        z[i] += bias[i];
    }
    
    // Dropout
    if (layer_type == Layer1DType::Dropout && is_training) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < z.size(); i++) {
            dropout_mask[i] = (dist(gen) > dropout_rate);
            if (!dropout_mask[i]) {
                z[i] = 0.0;
            } else {
                z[i] /= (1.0 - dropout_rate);
            }
        }
    }
    
    // 激活函数
    output = Activation::forward(z, activation);
    return output;
}

std::vector<double> Layer1D::backward(const std::vector<double>& gradient, Optimizer& optimizer) {
    // 计算激活函数的导数
    std::vector<double> delta = Activation::backward(output, activation);
    
    // Hadamard积
    for (size_t i = 0; i < delta.size(); i++) {
        delta[i] *= gradient[i];
    }
    
    // 如果是Dropout层，应用mask
    if (layer_type == Layer1DType::Dropout && is_training) {
        for (size_t i = 0; i < delta.size(); i++) {
            if (!dropout_mask[i]) {
                delta[i] = 0.0;
            } else {
                delta[i] /= (1.0 - dropout_rate);
            }
        }
    }
    
    // 计算权重梯度
    std::vector<std::vector<double>> weight_gradients(weights.size(), 
        std::vector<double>(weights[0].size()));
    
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[0].size(); j++) {
            weight_gradients[i][j] = delta[i] * input[j];
        }
    }
    
    // 计算输入的梯度（用于反向传播到前一层）
    std::vector<double> input_gradient(input.size(), 0.0);
    for (size_t i = 0; i < weights[0].size(); i++) {
        for (size_t j = 0; j < weights.size(); j++) {
            input_gradient[i] += weights[j][i] * delta[j];
        }
    }
    
    // 更新参数
    optimizer.update(weights, bias, weight_gradients, delta);
    
    return input_gradient;
}