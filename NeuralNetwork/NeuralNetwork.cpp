#include <iostream>
#include <vector>
#include "../utils/Layers/include/Layer1D.h"
#include "../utils/Common/include/Activation.h"
#include "../utils/Common/include/Loss.h"
#include "../utils/Common/include/Optimizer.h"
#include "../utils/MatrixOps/MatrixOps.h"

class NeuralNetwork {
private:
    std::vector<Layer1D*> layers;
    std::vector<int> topology;
    int batch_size;
    double learning_rate;
    std::vector<double> input;  // 存储当前输入用于反向传播
    SGD optimizer;

public:
    NeuralNetwork(const std::vector<int>& topology, 
                  int batch_size = 16,
                  double lr = 0.01)
        : topology(topology),
          batch_size(batch_size),
          learning_rate(lr),
          optimizer(lr) {
        
        if(topology.size() < 2) {
            std::cout << "Error: Network must have at least input and output layers" << std::endl;
            return;
        }
        
        std::cout << "Creating network with topology: ";
        for(int size : topology) {
            std::cout << size << " ";
        }
        std::cout << std::endl;
        
        // 创建层
        for(size_t i = 0; i < topology.size() - 1; i++) {
            // 最后一层使用Sigmoid激活函数，其他层使用ReLU
            ActivationFunction act = (i == topology.size() - 2) ? 
                                   ActivationFunction::SIGMOID : 
                                   ActivationFunction::RELU;
            
            // 创建线性层
            layers.push_back(new Layer1D(
                "layer_" + std::to_string(i),
                topology[i],
                topology[i + 1],
                Layer1DType::Linear,
                act
            ));
            
            // 在隐藏层之后添加Dropout层（除了输出层）
            if(i < topology.size() - 2) {
                layers.push_back(new Layer1D(
                    "dropout_" + std::to_string(i),
                    topology[i + 1],
                    topology[i + 1],
                    Layer1DType::Dropout,
                    act,
                    0.2  // dropout率
                ));
            }
        }
    }

    ~NeuralNetwork() {
        for(auto layer : layers) {
            delete layer;
        }
    }

    std::vector<double> forward(const std::vector<double>& input_data) {
        input = input_data;  // 存储输入
        std::vector<double> current = input_data;
        
        for(auto layer : layers) {
            current = layer->forward(current);
        }
        
        return current;
    }

    void backward(const std::vector<double>& output,
                 const std::vector<double>& target,
                 const LossFunction& loss_type = LossFunction::MSE) {
        
        // 计算损失函数的导数
        std::vector<double> gradient = Loss::derivative(output, target, loss_type);
        
        // 反向传播
        for(int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers[i]->backward(gradient, optimizer);
        }
    }

    void train(const std::vector<std::vector<double>>& train_data,
              const std::vector<std::vector<double>>& targets,
              int epochs,
              int batch_size) {
        
        int num_batches = (train_data.size() + batch_size - 1) / batch_size;
        
        for(int epoch = 0; epoch < epochs; epoch++) {
            double epoch_loss = 0.0;
            
            // 设置训练模式
            for(auto layer : layers) {
                layer->setTrainingMode(true);
            }
            
            for(int batch = 0; batch < num_batches; batch++) {
                double batch_loss = 0.0;
                int start_idx = batch * batch_size;
                int end_idx = std::min(start_idx + batch_size, (int)train_data.size());
                
                for(int i = start_idx; i < end_idx; i++) {
                    // 前向传播
                    std::vector<double> output = forward(train_data[i]);
                    
                    // 反向传播
                    backward(output, targets[i]);
                    
                    // 计算损失
                    batch_loss += Loss::compute(output, targets[i], LossFunction::MSE);
                }
                
                batch_loss /= (end_idx - start_idx);
                epoch_loss += batch_loss;
            }
            
            epoch_loss /= num_batches;
            
            if(epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << epoch_loss << std::endl;
            }
        }
    }
};

int main() {
    std::vector<int> topology = {2, 8, 8, 1};
    NeuralNetwork nn(topology, 4, 0.1);
    
    // XOR训练数据
    std::vector<std::vector<double>> train_data = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    
    std::vector<std::vector<double>> targets = {
        {0}, {1}, {1}, {0}
    };
    
    std::cout << "Training started..." << std::endl;
    nn.train(train_data, targets, 10000, 4);
    
    std::cout << "\nTesting the network:" << std::endl;
    for(size_t i = 0; i < train_data.size(); i++) {
        std::vector<double> output = nn.forward(train_data[i]);
        std::cout << train_data[i][0] << " XOR " << train_data[i][1] 
                 << " = " << output[0] 
                 << " (expected: " << targets[i][0] << ")" << std::endl;
    }
    
    return 0;
}