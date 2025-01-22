#include "../include/Loss.h"
#include <cmath>
#include <limits>

namespace Loss {
    double compute(const std::vector<double>& output,
                  const std::vector<double>& target,
                  const LossFunction& loss_type) {
        double loss = 0.0;
        
        switch(loss_type) {
            case LossFunction::MSE:
                for(size_t i = 0; i < output.size(); i++) {
                    loss += std::pow(output[i] - target[i], 2);
                }
                return loss / output.size();
                
            case LossFunction::RMSE:
                for(size_t i = 0; i < output.size(); i++) {
                    loss += std::pow(output[i] - target[i], 2);
                }
                return std::sqrt(loss / output.size());
                
            case LossFunction::CROSS_ENTROPY:
                for(size_t i = 0; i < output.size(); i++) {
                    // 添加数值稳定性处理
                    double epsilon = 1e-15;
                    double y_pred = std::max(std::min(output[i], 1.0 - epsilon), epsilon);
                    loss -= target[i] * std::log(y_pred);
                }
                return loss;
        }
        return loss;
    }

    std::vector<double> derivative(const std::vector<double>& output,
                                 const std::vector<double>& target,
                                 const LossFunction& loss_type) {
        std::vector<double> grad(output.size());
        
        switch(loss_type) {
            case LossFunction::MSE:
                for(size_t i = 0; i < output.size(); i++) {
                    grad[i] = 2.0 * (output[i] - target[i]) / output.size();
                }
                break;
                
            case LossFunction::RMSE: {
                double mse = 0.0;
                for(size_t i = 0; i < output.size(); i++) {
                    mse += std::pow(output[i] - target[i], 2);
                }
                mse /= output.size();
                double rmse = std::sqrt(mse);
                
                for(size_t i = 0; i < output.size(); i++) {
                    grad[i] = (output[i] - target[i]) / (output.size() * rmse);
                }
                break;
            }
                
            case LossFunction::CROSS_ENTROPY:
                // 对于Softmax + Cross Entropy，导数就是 output - target
                for(size_t i = 0; i < output.size(); i++) {
                    grad[i] = output[i] - target[i];
                }
                break;
        }
        return grad;
    }

    std::string toString(const LossFunction& loss_type) {
        switch(loss_type) {
            case LossFunction::MSE: return "MSE";
            case LossFunction::RMSE: return "RMSE";
            case LossFunction::CROSS_ENTROPY: return "Cross Entropy";
            default: return "UNKNOWN";
        }
    }
}