#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <numeric>  // 为std::iota添加
#include "../utils/Layers/include/Layer3D.h"
#include "../utils/Common/include/Loss.h"
#include "../utils/Common/include/Optimizer.h"

class CNN {
private:
    std::vector<Layer3D*> layers;
    std::vector<int> layerDims;  // 每层的输出通道数
    int inputHeight;
    int inputWidth;
    int inputChannels;
    int numClasses;
    double learningRate;
    SGD optimizer;

    // Helper function to calculate accuracy
    double calculateAccuracy(const std::vector<double>& prediction, const std::vector<double>& label) {
        int predicted_class = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
        int true_class = std::max_element(label.begin(), label.end()) - label.begin();
        return (predicted_class == true_class) ? 1.0 : 0.0;
    }

public:
    CNN(int inputHeight, 
        int inputWidth, 
        int inputChannels,
        int numClasses,
        double lr = 0.01)
        : inputHeight(inputHeight),
          inputWidth(inputWidth),
          inputChannels(inputChannels),
          numClasses(numClasses),
          learningRate(lr),
          optimizer(lr) {
        
        std::cout << "\n=== Network Architecture ===\n" << std::endl;
        std::cout << "Input dimensions: " << inputHeight << "x" 
                  << inputWidth << "x" << inputChannels << std::endl;
    }

    ~CNN() {
        for(auto layer : layers) {
            delete layer;
        }
    }

    // Forward pass
    std::vector<std::vector<std::vector<double>>> forward(
        const std::vector<std::vector<std::vector<double>>>& input) {
        
        std::vector<std::vector<std::vector<double>>> current = input;
        
        std::cout << "\nForward Pass Debug:" << std::endl;
        std::cout << "Input shape: " << current.size() << "x" 
                  << current[0].size() << "x" << current[0][0].size() << std::endl;
        
        for(size_t i = 0; i < layers.size(); i++) {
            try {
                std::cout << "Processing layer " << i + 1 << "..." << std::endl;
                current = layers[i]->forward(current);
                std::cout << "Layer " << i + 1 << " output shape: " << current.size() 
                          << "x" << current[0].size() << "x" << current[0][0].size() << std::endl;
                
                // Print some statistics
                double min_val = std::numeric_limits<double>::max();
                double max_val = std::numeric_limits<double>::lowest();
                for(const auto& channel : current) {
                    for(const auto& row : channel) {
                        for(const auto& val : row) {
                            min_val = std::min(min_val, val);
                            max_val = std::max(max_val, val);
                        }
                    }
                }
                std::cout << "  Value range: [" << min_val << ", " << max_val << "]" << std::endl;
            }
            catch(const std::exception& e) {
                std::cerr << "Error in layer " << i + 1 << ": " << e.what() << std::endl;
                throw;
            }
        }
        
        return current;
    }

    // Backward pass
    void backward(const std::vector<std::vector<std::vector<double>>>& gradient) {
        std::vector<std::vector<std::vector<double>>> current_gradient = gradient;
        
        for(int i = layers.size() - 1; i >= 0; i--) {
            current_gradient = layers[i]->backward(current_gradient);
        }
    }

    // Training function
    void train(const std::vector<std::vector<std::vector<std::vector<double>>>>& trainData,
              const std::vector<std::vector<double>>& labels,
              int epochs,
              int batchSize) {
        
        std::cout << "=== Training Configuration ===\n" << std::endl;
        std::cout << "Total samples: " << trainData.size() << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Learning rate: " << learningRate << std::endl;
        std::cout << "\n=== Training Progress ===\n" << std::endl;
        
        int numSamples = trainData.size();
        int numBatches = (numSamples + batchSize - 1) / batchSize;
        
        // For calculating average metrics
        std::vector<double> epoch_losses;
        std::vector<double> epoch_accuracies;
        
        for(int epoch = 0; epoch < epochs; epoch++) {
            double epochLoss = 0.0;
            double epochAccuracy = 0.0;
            
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
            
            // 创建一个样本索引的向量，用于随机打乱
            std::vector<int> indices(numSamples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);  // 使用std::shuffle替代random_shuffle
            
            for(int batch = 0; batch < numBatches; batch++) {
                int startIdx = batch * batchSize;
                int endIdx = std::min(startIdx + batchSize, numSamples);
                int currentBatchSize = endIdx - startIdx;
                double batchLoss = 0.0;
                double batchAccuracy = 0.0;
                
                // 累积梯度
                for(auto layer : layers) {
                    layer->clearGradients();  // 使用clearGradients替代resetGradients
                }
                
                // 处理当前batch中的所有样本
                for(int i = startIdx; i < endIdx; i++) {
                    int idx = indices[i];  // 使用打乱后的索引
                    try {
                        if(batch == 0 && i == startIdx) {  // 只在第一个样本时打印详细信息
                            std::cout << "\nProcessing first sample of batch..." << std::endl;
                        }
                        
                        // Forward pass
                        std::vector<std::vector<std::vector<double>>> output = forward(trainData[idx]);
                        
                        // Flatten output
                        std::vector<double> flattened_output;
                        for(const auto& channel : output) {
                            for(const auto& row : channel) {
                                flattened_output.insert(flattened_output.end(), row.begin(), row.end());
                            }
                        }
                        
                        // 计算损失和准确率
                        double loss = Loss::compute(flattened_output, labels[idx], LossFunction::CROSS_ENTROPY);
                        batchLoss += loss;
                        batchAccuracy += calculateAccuracy(flattened_output, labels[idx]);
                        
                        // 计算梯度
                        std::vector<double> loss_gradient = Loss::derivative(flattened_output, labels[idx], LossFunction::CROSS_ENTROPY);
                        
                        // Reshape gradient to 3D
                        int out_channels = output.size();
                        int out_height = output[0].size();
                        int out_width = output[0][0].size();
                        
                        std::vector<std::vector<std::vector<double>>> gradient(
                            out_channels,
                            std::vector<std::vector<double>>(
                                out_height,
                                std::vector<double>(out_width)
                            )
                        );
                        
                        int idx_g = 0;
                        for(int c = 0; c < out_channels; ++c) {
                            for(int h = 0; h < out_height; ++h) {
                                for(int w = 0; w < out_width; ++w) {
                                    gradient[c][h][w] = loss_gradient[idx_g++];
                                }
                            }
                        }
                        
                        // Backward pass (累积梯度)
                        backward(gradient);
                    }
                    catch(const std::exception& e) {
                        std::cerr << "Error processing sample " << idx + 1 << ": " << e.what() << std::endl;
                        throw;
                    }
                }
                
                // 更新参数（使用平均梯度）
                for(auto layer : layers) {
                    layer->multiplyGradients(1.0 / currentBatchSize);  // 使用multiplyGradients替代scaleGradients
                    layer->updateParameters(optimizer);
                }
                
                batchLoss /= currentBatchSize;
                batchAccuracy /= currentBatchSize;
                epochLoss += batchLoss;
                epochAccuracy += batchAccuracy;
                
                // 每处理几个batch打印一次进度
                if ((batch + 1) % 5 == 0 || batch == numBatches - 1) {
                    std::cout << "  Batch " << batch + 1 << "/" << numBatches 
                              << " - Loss: " << std::fixed << std::setprecision(4) << batchLoss
                              << " - Accuracy: " << std::setprecision(2) << (batchAccuracy * 100) << "%" 
                              << std::endl;
                }
            }
            
            epochLoss /= numBatches;
            epochAccuracy /= numBatches;
            epoch_losses.push_back(epochLoss);
            epoch_accuracies.push_back(epochAccuracy);
            
            std::cout << "\nEpoch " << epoch + 1 << " Summary:" << std::endl;
            std::cout << "  Average Loss: " << std::fixed << std::setprecision(4) << epochLoss << std::endl;
            std::cout << "  Average Accuracy: " << std::setprecision(2) << (epochAccuracy * 100) << "%" << std::endl;
            
            if (epoch < epochs - 1) {
                std::cout << "\n" << std::string(50, '-') << "\n" << std::endl;
            }
        }
        
        // Print training summary
        std::cout << "\n=== Training Summary ===\n" << std::endl;
        std::cout << "Initial Loss: " << std::fixed << std::setprecision(4) << epoch_losses.front() << std::endl;
        std::cout << "Final Loss: " << epoch_losses.back() << std::endl;
        std::cout << "Initial Accuracy: " << std::setprecision(2) << (epoch_accuracies.front() * 100) << "%" << std::endl;
        std::cout << "Final Accuracy: " << (epoch_accuracies.back() * 100) << "%" << std::endl;
    }

    // Prediction function
    std::vector<double> predict(const std::vector<std::vector<std::vector<std::vector<double>>>>& input) {
        std::vector<std::vector<std::vector<double>>> output = forward(input[0]);
        
        // Flatten output
        std::vector<double> flattened_output;
        for(const auto& channel : output) {
            for(const auto& row : channel) {
                flattened_output.insert(flattened_output.end(), row.begin(), row.end());
            }
        }
        return flattened_output;
    }

    void addLayer(Layer3D* layer) {
        layers.push_back(layer);
        
        // 打印层信息
        std::string layerType;
        switch(layer->type) {
            case Layer3DType::Conv3D:
                std::cout << "Layer " << layers.size() << ": Convolution" << std::endl;
                break;
            case Layer3DType::MaxPool3D:
                std::cout << "Layer " << layers.size() << ": Max Pooling" << std::endl;
                break;
            case Layer3DType::Linear3D:
                std::cout << "Layer " << layers.size() << ": Fully Connected" << std::endl;
                break;
            default:
                std::cout << "Layer " << layers.size() << ": Unknown Type" << std::endl;
        }
        
        // 初始化层
        layer->initialize();
    }
};

// Test code
int main() {
    try {
        std::cout << "\n====== CNN Training Demo ======\n" << std::endl;
        
        // 生成简单的训练数据
        std::vector<std::vector<std::vector<std::vector<double>>>> trainingData;
        std::vector<std::vector<double>> trainingLabels;

        // 设置数据集大小
        const int numSamples = 100;  // 减少样本数量
        const int imageSize = 28;
        const int numClasses = 2;    // 改为二分类

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 0.1);

        // 生成训练数据
        for(int i = 0; i < numSamples; ++i) {
            // 创建一个空白图像
            std::vector<std::vector<std::vector<double>>> sample(
                1,  // 单通道
                std::vector<std::vector<double>>(
                    imageSize,
                    std::vector<double>(imageSize, 0.0)
                )
            );
            
            // 生成标签 (0 或 1)
            int label = i % 2;
            std::vector<double> oneHot(numClasses, 0.0);
            oneHot[label] = 1.0;
            
            // 根据类别生成不同的图案
            if(label == 0) {
                // 类别0：在图像中心绘制一个更大的方形
                int center = imageSize / 2;
                int size = 12;  // 增大方形尺寸
                for(int h = center - size/2; h < center + size/2; ++h) {
                    for(int w = center - size/2; w < center + size/2; ++w) {
                        sample[0][h][w] = 1.0;  // 使用固定值1.0，不加噪声
                    }
                }
            } else {
                // 类别1：在图像中心绘制一个更大的圆形
                int center = imageSize / 2;
                int radius = 6;  // 增大圆形半径
                for(int h = 0; h < imageSize; ++h) {
                    for(int w = 0; w < imageSize; ++w) {
                        double dist = std::sqrt(std::pow(h - center, 2) + std::pow(w - center, 2));
                        if(dist <= radius) {
                            sample[0][h][w] = 1.0;  // 使用固定值1.0，不加噪声
                        }
                    }
                }
            }
            
            // 减少背景噪声
            for(int h = 0; h < imageSize; ++h) {
                for(int w = 0; w < imageSize; ++w) {
                    if(sample[0][h][w] == 0.0) {
                        sample[0][h][w] = std::abs(dis(gen)) * 0.05;  // 减小背景噪声
                    }
                }
            }
            
            trainingData.push_back(sample);
            trainingLabels.push_back(oneHot);
        }

        // 创建CNN实例
        CNN cnn(28, 28, 1, 2, 0.1);  // 增大学习率到0.1

        // 添加层 - 简化网络结构
        cnn.addLayer(new Layer3D(
            "conv1",
            Layer3DType::Conv3D,
            ActivationFunction::RELU,
            5,                          // 增大卷积核到5x5
            1,                          // 输入通道
            8,                          // 增加特征图数量
            1,                          // stride
            2,                          // 增加padding
            0,                          // poolSize (不使用)
            28,                         // 输入高度
            28                          // 输入宽度
        ));

        cnn.addLayer(new Layer3D(
            "pool1",
            Layer3DType::MaxPool3D,
            ActivationFunction::RELU,
            2,                          // kernel size
            8,                          // 输入通道
            8,                          // 输出通道
            2,                          // stride
            0,                          // padding
            2,                          // poolSize
            28,                         // 输入高度
            28                          // 输入宽度
        ));

        cnn.addLayer(new Layer3D(
            "fc1",
            Layer3DType::Linear3D,
            ActivationFunction::SOFTMAX,
            1,                          // kernel size (不使用)
            8,                          // 输入通道
            numClasses,                 // 输出通道
            1,                          // stride (不使用)
            0,                          // padding (不使用)
            0,                          // poolSize (不使用)
            14,                         // 输入高度
            14                          // 输入宽度
        ));

        // 训练参数
        const int epochs = 20;          // 增加训练轮数
        const int batchSize = 5;        // 减小batch size

        // 开始训练
        std::cout << "\n=== Starting Training ===\n" << std::endl;
        cnn.train(trainingData, trainingLabels, epochs, batchSize);

        // 在验证集上评估
        std::cout << "\n=== Validation Set Evaluation ===\n" << std::endl;
        int correctPredictions = 0;
        double totalLoss = 0.0;
        
        for(size_t i = 0; i < trainingData.size(); ++i) {
            auto prediction = cnn.predict({trainingData[i]});  // 注意这里需要包装成vector
            
            // 计算损失
            totalLoss += Loss::compute(prediction, trainingLabels[i], LossFunction::CROSS_ENTROPY);
            
            // 计算准确率
            int predicted_class = std::max_element(prediction.begin(), prediction.end()) - prediction.begin();
            int true_class = std::max_element(trainingLabels[i].begin(), trainingLabels[i].end()) - trainingLabels[i].begin();
            if(predicted_class == true_class) {
                correctPredictions++;
            }
        }
        
        double avgLoss = totalLoss / trainingData.size();
        double accuracy = static_cast<double>(correctPredictions) / trainingData.size() * 100.0;
        
        std::cout << "Validation Results:" << std::endl;
        std::cout << "  Average Loss: " << std::fixed << std::setprecision(4) << avgLoss << std::endl;
        std::cout << "  Accuracy: " << std::setprecision(2) << accuracy << "%" << std::endl;
        
    }
    catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "\nUnknown error occurred" << std::endl;
    }
    
    return 0;
}
