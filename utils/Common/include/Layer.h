#pragma once
#include <vector>
#include "Activation.h"
#include "Optimizer.h"

enum class LayerType {
    Linear,
    Dropout
};

class Layer {
protected:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> outputs;
    std::vector<double> deltas;
    std::vector<double> z;
    ActivationFunction activation;
    LayerType layer_type;
    double dropout_rate;
    std::vector<bool> dropout_mask;
    bool is_training;

public:
    Layer(int input_size, int output_size,
          ActivationFunction act_type, LayerType layer_type);
    
    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& gradient, Optimizer& optimizer);
    
    // getters and setters
    void setTrainingMode(bool is_train) { is_training = is_train; }
    const std::vector<double>& getOutput() const { return outputs; }
};