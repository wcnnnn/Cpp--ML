#include "../include/Layer3D.h"
#include "../../Common/include/Activation.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

void Layer3D::initialize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if(type == Layer3DType::Conv3D || type == Layer3DType::Linear3D) {
        // Xavier初始化
        double limit;
        if(type == Layer3DType::Conv3D) {
            limit = std::sqrt(2.0 / (inputChannels * kernelSize * kernelSize));
        } else {  // Linear3D
            int fan_in = inputChannels * inputHeight * inputWidth;
            limit = std::sqrt(2.0 / fan_in);
        }
        
        std::uniform_real_distribution<> dis(-limit, limit);
        
        if(type == Layer3DType::Conv3D) {
            kernels = std::vector<std::vector<std::vector<double>>>(
                outputChannels,
                std::vector<std::vector<double>>(
                    inputChannels,
                    std::vector<double>(kernelSize * kernelSize)
                )
            );
        } else {  // Linear3D
            int inputSize = inputChannels * inputHeight * inputWidth;
            kernels = std::vector<std::vector<std::vector<double>>>(
                outputChannels,
                std::vector<std::vector<double>>(
                    1,
                    std::vector<double>(inputSize)
                )
            );
        }
        
        for(auto& och : kernels) {
            for(auto& ich : och) {
                for(auto& w : ich) {
                    w = dis(gen);
                }
            }
        }
        
        // 初始化偏置为小值
        bias = std::vector<double>(outputChannels, 0.0);
    }
}

std::vector<std::vector<std::vector<double>>> Layer3D::forward(
    const std::vector<std::vector<std::vector<double>>>& input) {
    this->input = input;
    
    // 确保梯度存储的大小与输入匹配
    gradients = std::vector<std::vector<std::vector<double>>>(
        input.size(),
        std::vector<std::vector<double>>(
            input[0].size(),
            std::vector<double>(input[0][0].size(), 0.0)
        )
    );
    
    switch(type) {
        case Layer3DType::Conv3D: {
            // 执行3D卷积
            output = CNNOps::Convolution::forward3D(input, kernels, stride, padding);
            
            // 添加偏置并应用激活函数
            for(size_t oc = 0; oc < output.size(); ++oc) {
                for(size_t h = 0; h < output[oc].size(); ++h) {
                    for(size_t w = 0; w < output[oc][h].size(); ++w) {
                        output[oc][h][w] += bias[oc];
                        std::vector<double> temp = {output[oc][h][w]};
                        temp = Activation::forward(temp, activation);
                        output[oc][h][w] = temp[0];
                    }
                }
            }
            break;
        }
            
        case Layer3DType::MaxPool3D: {
            output = CNNOps::Pooling::maxPool3D(input, poolSize, stride);
            break;
        }
            
        case Layer3DType::AvgPool3D: {
            output = CNNOps::Pooling::avgPool3D(input, poolSize, stride);
            break;
        }
            
        case Layer3DType::Linear3D: {
            // 更新输入维度
            inputHeight = input[0].size();
            inputWidth = input[0][0].size();
            
            // 展平输入
            std::vector<double> flattened_input;
            flattened_input.reserve(inputChannels * inputHeight * inputWidth);
            for(const auto& channel : input) {
                for(const auto& row : channel) {
                    flattened_input.insert(flattened_input.end(), row.begin(), row.end());
                }
            }
            
            // 执行线性变换
            output = std::vector<std::vector<std::vector<double>>>(
                1,  // 输出只有一个通道
                std::vector<std::vector<double>>(
                    1,  // 输出高度为1
                    std::vector<double>(outputChannels, 0.0)  // 输出宽度为类别数
                )
            );
            
            // 计算输出
            std::vector<double>& out = output[0][0];
            
            // 线性变换
            for(int oc = 0; oc < outputChannels; ++oc) {
                double sum = 0.0;
                for(size_t i = 0; i < flattened_input.size(); ++i) {
                    sum += kernels[oc][0][i] * flattened_input[i];
                }
                out[oc] = sum + bias[oc];
            }
            
            // 如果是Softmax激活，进行数值稳定性处理
            if(activation == ActivationFunction::SOFTMAX) {
                // 找到最大值
                double max_val = *std::max_element(out.begin(), out.end());
                
                // 减去最大值并计算指数和
                double sum_exp = 0.0;
                std::vector<double> exp_values(outputChannels);
                
                for(int oc = 0; oc < outputChannels; ++oc) {
                    exp_values[oc] = std::exp(out[oc] - max_val);
                    sum_exp += exp_values[oc];
                }
                
                // 归一化
                if(sum_exp > 1e-10) {  // 避免除以非常小的数
                    for(int oc = 0; oc < outputChannels; ++oc) {
                        out[oc] = exp_values[oc] / sum_exp;
                    }
                } else {
                    // 如果所有值都非常小，使用均匀分布
                    for(int oc = 0; oc < outputChannels; ++oc) {
                        out[oc] = 1.0 / outputChannels;
                    }
                }
            } else {
                // 其他激活函数
                out = Activation::forward(out, activation);
            }
            
            break;
        }
            
        default:
            throw std::runtime_error("Unsupported layer type");
    }
    
    return output;
}

std::vector<std::vector<std::vector<double>>> Layer3D::backward(
    const std::vector<std::vector<std::vector<double>>>& gradient) {
    
    std::vector<std::vector<std::vector<double>>> nextGradient;
    
    switch(type) {
        case Layer3DType::Conv3D: {
            // 计算激活函数的导数
            gradients = gradient;
            for(size_t c = 0; c < gradients.size(); ++c) {
                for(size_t h = 0; h < gradients[c].size(); ++h) {
                    for(size_t w = 0; w < gradients[c][h].size(); ++w) {
                        std::vector<double> temp = {output[c][h][w]};
                        std::vector<double> grad_temp = Activation::backward(temp, activation);
                        gradients[c][h][w] *= grad_temp[0];
                    }
                }
            }
            
            // 计算卷积的反向传播
            nextGradient = CNNOps::Convolution::backward3D(gradients, kernels, stride, padding);
            break;
        }
            
        case Layer3DType::MaxPool3D: {
            nextGradient = CNNOps::Pooling::maxPoolBackward3D(gradient, input, poolSize, stride);
            break;
        }
            
        case Layer3DType::AvgPool3D: {
            nextGradient = CNNOps::Pooling::avgPoolBackward3D(gradient, poolSize, stride);
            break;
        }
            
        case Layer3DType::Linear3D: {
            // 计算激活函数的导数
            std::vector<double> output_grad = gradient[0][0];
            
            if(activation == ActivationFunction::SOFTMAX) {
                // 对于交叉熵损失，直接使用(output - target)作为梯度
                // output_grad已经是(output - target)
                // 不需要额外计算Softmax的导数
            } else {
                std::vector<double> activation_grad = Activation::backward(output[0][0], activation);
                // Hadamard积
                for(size_t i = 0; i < output_grad.size(); ++i) {
                    output_grad[i] *= activation_grad[i];
                }
            }
            
            // 展平输入
            std::vector<double> flattened_input;
            int total_input_size = input.size() * input[0].size() * input[0][0].size();
            flattened_input.reserve(total_input_size);
            
            for(const auto& channel : input) {
                for(const auto& row : channel) {
                    flattened_input.insert(flattened_input.end(), row.begin(), row.end());
                }
            }
            
            // 初始化权重梯度
            gradients = std::vector<std::vector<std::vector<double>>>(
                outputChannels,
                std::vector<std::vector<double>>(
                    1,  // 一个通道
                    std::vector<double>(total_input_size, 0.0)  // 使用实际的输入大小
                )
            );
            
            // 计算权重的梯度
            for(int oc = 0; oc < outputChannels; ++oc) {
                for(size_t i = 0; i < flattened_input.size(); ++i) {
                    gradients[oc][0][i] = output_grad[oc] * flattened_input[i];
                }
            }
            
            // 计算输入的梯度
            nextGradient = std::vector<std::vector<std::vector<double>>>(
                input.size(),
                std::vector<std::vector<double>>(
                    input[0].size(),
                    std::vector<double>(input[0][0].size(), 0.0)
                )
            );
            
            // 重塑梯度回3D形状
            int idx = 0;
            for(size_t c = 0; c < input.size(); ++c) {
                for(size_t h = 0; h < input[0].size(); ++h) {
                    for(size_t w = 0; w < input[0][0].size(); ++w) {
                        double grad_sum = 0.0;
                        for(int oc = 0; oc < outputChannels; ++oc) {
                            grad_sum += output_grad[oc] * kernels[oc][0][idx];
                        }
                        nextGradient[c][h][w] = grad_sum;
                        idx++;
                    }
                }
            }
            
            // 打印调试信息
            std::cout << "Linear layer backward pass:" << std::endl;
            std::cout << "  Input shape: " << input.size() << "x" 
                      << input[0].size() << "x" << input[0][0].size() << std::endl;
            std::cout << "  Output gradient shape: " << gradient.size() << "x"
                      << gradient[0].size() << "x" << gradient[0][0].size() << std::endl;
            std::cout << "  Weight gradients shape: " << gradients.size() << "x"
                      << gradients[0].size() << "x" << gradients[0][0].size() << std::endl;
            std::cout << "  Max output gradient: " << *std::max_element(output_grad.begin(), output_grad.end()) << std::endl;
            std::cout << "  Min output gradient: " << *std::min_element(output_grad.begin(), output_grad.end()) << std::endl;
            
            break;
        }
            
        default:
            throw std::runtime_error("Unsupported layer type");
    }
    
    return nextGradient;
}

void Layer3D::updateParameters(Optimizer& optimizer) {
    if(type == Layer3DType::Conv3D || type == Layer3DType::Linear3D) {
        // 梯度裁剪
        double maxGradNorm = 5.0;
        double gradNorm = 0.0;
        
        // 计算梯度范数
        for(const auto& och : gradients) {
            for(const auto& ich : och) {
                for(const auto& grad : ich) {
                    gradNorm += grad * grad;
                }
            }
        }
        gradNorm = std::sqrt(gradNorm);
        
        // 如果梯度范数过大，进行裁剪
        if(gradNorm > maxGradNorm) {
            double scale = maxGradNorm / gradNorm;
            for(auto& och : gradients) {
                for(auto& ich : och) {
                    for(auto& grad : ich) {
                        grad *= scale;
                    }
                }
            }
        }
        
        // 计算偏置梯度
        std::vector<double> biasGradients(outputChannels, 0.0);
        for(int oc = 0; oc < outputChannels; ++oc) {
            if(type == Layer3DType::Conv3D) {
                // 对于卷积层，累加所有位置的梯度
                for(size_t h = 0; h < gradients[oc].size(); ++h) {
                    for(size_t w = 0; w < gradients[oc][h].size(); ++w) {
                        biasGradients[oc] += gradients[oc][h][w];
                    }
                }
                // 计算平均值
                biasGradients[oc] /= (gradients[oc].size() * gradients[oc][0].size());
            } else {  // Linear3D
                // 对于线性层，直接使用第一个梯度
                biasGradients[oc] = gradients[oc][0][0];
            }
        }
        
        // 更新参数
        if(type == Layer3DType::Linear3D) {
            // Linear层使用较小的学习率
            optimizer.setLearningRate(optimizer.getLearningRate() * 0.1);
        }
        
        // 批量更新参数
        optimizer.update3D(kernels, bias, gradients, biasGradients);
        
        if(type == Layer3DType::Linear3D) {
            // 恢复学习率
            optimizer.setLearningRate(optimizer.getLearningRate() * 10);
        }
    }
}
