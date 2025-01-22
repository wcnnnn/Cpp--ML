#include "../include/Activation.h"
#include <cmath>
#include <algorithm>

namespace Activation {
    std::vector<double> forward(const std::vector<double>& input, 
                              const ActivationFunction& func_type) {
        std::vector<double> result(input.size());
        
        switch(func_type) {
            case ActivationFunction::SOFTMAX: {
                double max_val = *std::max_element(input.begin(), input.end());
                double sum = 0.0;
                for(size_t i = 0; i < input.size(); i++) {
                    result[i] = std::exp(input[i] - max_val);
                    sum += result[i];
                }
                for(size_t i = 0; i < input.size(); i++) {
                    result[i] /= sum;
                }
                break;
            }
            case ActivationFunction::SIGMOID:
                for(size_t i = 0; i < input.size(); i++) {
                    result[i] = 1.0 / (1.0 + std::exp(-input[i]));
                }
                break;
            case ActivationFunction::RELU:
                for(size_t i = 0; i < input.size(); i++) {
                    result[i] = std::max(0.0, input[i]);
                }
                break;
            case ActivationFunction::TANH:
                for(size_t i = 0; i < input.size(); i++) {
                    result[i] = std::tanh(input[i]);
                }
                break;
        }
        return result;
    }

    std::vector<double> backward(const std::vector<double>& input, 
                               const ActivationFunction& func_type) {
        std::vector<double> derivative(input.size());
        
        switch(func_type) {
            case ActivationFunction::SOFTMAX:
                // Softmax的导数与交叉熵损失一起使用时会简化
                return input;
            case ActivationFunction::SIGMOID:
                for(size_t i = 0; i < input.size(); i++) {
                    derivative[i] = input[i] * (1.0 - input[i]);
                }
                break;
            case ActivationFunction::RELU:
                for(size_t i = 0; i < input.size(); i++) {
                    derivative[i] = input[i] > 0 ? 1.0 : 0.0;
                }
                break;
            case ActivationFunction::TANH:
                for(size_t i = 0; i < input.size(); i++) {
                    derivative[i] = 1.0 - input[i] * input[i];
                }
                break;
        }
        return derivative;
    }

    std::string toString(const ActivationFunction& func_type) {
        switch(func_type) {
            case ActivationFunction::SIGMOID: return "SIGMOID";
            case ActivationFunction::RELU: return "RELU";
            case ActivationFunction::TANH: return "TANH";
            case ActivationFunction::SOFTMAX: return "SOFTMAX";
            default: return "UNKNOWN";
        }
    }
}