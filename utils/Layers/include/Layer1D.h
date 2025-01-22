#pragma once
#include "Layer.h"
#include <random>

enum class Layer1DType {
    Linear,
    Dropout
};

class Layer1D : public Layer {
protected:
    Layer1DType layer_type;
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> output;
    std::vector<double> input;  // 存储输入用于反向传播
    std::vector<double> z;      // 存储激活前的值
    
    // Dropout特定参数
    double dropout_rate;
    std::vector<bool> dropout_mask;

public:
    Layer1D(const std::string& name, 
            int input_size, 
            int output_size,
            Layer1DType type,
            ActivationFunction act = ActivationFunction::RELU,
            double dropout_rate = 0.2);

    virtual void initialize() override;
    virtual std::vector<double> forward(const std::vector<double>& input);
    virtual std::vector<double> backward(const std::vector<double>& gradient, Optimizer& optimizer);

    // Getters
    const std::vector<double>& getOutput() const { return output; }
    const std::vector<std::vector<double>>& getWeights() const { return weights; }
    const std::vector<double>& getBias() const { return bias; }
};