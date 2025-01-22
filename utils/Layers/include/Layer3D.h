#pragma once
#include "Layer.h"
#include "../../CNNOps/CNNOps.h"

enum class Layer3DType {
    Conv3D,
    MaxPool3D,
    AvgPool3D,
    Linear3D
};

class Layer3D : public Layer {
public:
    Layer3DType type;
    
protected:
    std::vector<std::vector<std::vector<double>>> input;  // [channels][height][width]
    std::vector<std::vector<std::vector<double>>> output; // [channels][height][width]
    std::vector<std::vector<std::vector<double>>> gradients; // [channels][height][width]
    
    // 卷积层参数
    std::vector<std::vector<std::vector<double>>> kernels;  // [out_channels][in_channels][kernel_h*kernel_w]
    std::vector<double> bias;  // [out_channels]
    int kernelSize;
    int stride;
    int padding;
    int inputChannels;
    int outputChannels;
    
    // 池化层参数
    int poolSize;
    
    // 输入维度
    int inputHeight;
    int inputWidth;

    // 梯度
    std::vector<std::vector<std::vector<std::vector<double>>>> kernelGradients;
    std::vector<double> biasGradients;

public:
    Layer3D(const std::string& name,
            Layer3DType type,
            ActivationFunction act = ActivationFunction::RELU,
            int kernelSize = 3,
            int inputChannels = 1,
            int outputChannels = 1,
            int stride = 1,
            int padding = 0,
            int poolSize = 2,
            int inputHeight = 28,  // 默认MNIST图像大小
            int inputWidth = 28)   // 默认MNIST图像大小
        : Layer(name, act),
          type(type),
          kernelSize(kernelSize),
          stride(stride),
          padding(padding),
          inputChannels(inputChannels),
          outputChannels(outputChannels),
          poolSize(poolSize),
          inputHeight(inputHeight),
          inputWidth(inputWidth) {
        // 初始化梯度存储
        clearGradients();
    }

    void initialize() override;
    
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input);
        
    std::vector<std::vector<std::vector<double>>> backward(
        const std::vector<std::vector<std::vector<double>>>& gradient);
        
    void updateParameters(Optimizer& optimizer);

    // Getters
    const std::vector<std::vector<std::vector<double>>>& getOutput() const { return output; }
    Layer3DType getType() const { return type; }
    
    // 设置输入维度
    void setInputDimensions(int height, int width) {
        inputHeight = height;
        inputWidth = width;
    }

    // 清除梯度
    virtual void clearGradients() {
        using Vec4D = std::vector<std::vector<std::vector<std::vector<double>>>>;
        using Vec3D = std::vector<std::vector<std::vector<double>>>;
        using Vec2D = std::vector<std::vector<double>>;
        using Vec1D = std::vector<double>;

        kernelGradients = Vec4D(
            outputChannels,
            Vec3D(
                inputChannels,
                Vec2D(
                    kernelSize,
                    Vec1D(kernelSize, 0.0)
                )
            )
        );
        biasGradients = Vec1D(outputChannels, 0.0);
    }

    // 缩放梯度
    virtual void multiplyGradients(double factor) {
        for(auto& och : kernelGradients) {
            for(auto& ich : och) {
                for(auto& row : ich) {
                    for(auto& grad : row) {
                        grad *= factor;
                    }
                }
            }
        }
        for(auto& grad : biasGradients) {
            grad *= factor;
        }
    }
};