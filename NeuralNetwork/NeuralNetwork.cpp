#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "../utils/MatrixOps.h"
using namespace std;

enum class ActivationFunction {
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX
};

enum class Layers {  
    Linear,
    Dropout
};

enum class Optimizer {  
    SGD
};

enum class Loss_fuction {  
    MSE,
    RMSE
};

class SGD
{
private:
    double sgd_lr=0.01;
public:
    SGD(double lr =0.01):sgd_lr(lr){};
    void update(vector<vector<double>>& weights,vector<double>& bias,const vector<vector<double>>& weight_gradients,
    const vector<double>& bias_gradients){
        for (size_t i = 0; i < weights.size(); i++)
        {
            for (size_t j = 0; j < weights[0].size(); j++)
            {
                weights[i][j]=weights[i][j]-(sgd_lr*weight_gradients[i][j]);
            }
            
        }
        for (size_t i = 0; i < bias.size(); i++)
        {
            bias[i]=bias[i]-(sgd_lr*bias_gradients[i]);
        }
    }
};


double Loss(
    const vector<double>& error,
    const string& loss_function="MSE"){
    double loss = 0.0;
    if (loss_function=="MSE")
    {
        for (size_t i = 0; i < error.size(); i++)
        {
            loss+=pow(error[i],2);
        }
        return loss/error.size();
    }
    else if (loss_function=="RMSE")
    {
        for (size_t i = 0; i < error.size(); i++)
        {
            loss+=pow(error[i],2);
        }
        return sqrt(loss/error.size());
    }
    cout << "Warning: Unknown loss function, using MSE instead." << endl;
    for (size_t i = 0; i < error.size(); i++) {
        loss += pow(error[i], 2);
    }
    return loss/error.size();
}

vector<double> LossDerivative(
    const vector<double>& output, 
    const vector<double>& target,
    const string& loss_function="MSE") {
    
    vector<double> derivative(output.size());
    if (loss_function == "MSE") {
        // MSE导数：2(y_pred - y_true)/n
        for(size_t i = 0; i < output.size(); i++) {
            derivative[i] = 2.0 * (output[i] - target[i]) / output.size();
        }
    } else if (loss_function == "RMSE") {
        // RMSE导数：(y_pred - y_true)/(n * sqrt(MSE))
        double mse = 0.0;
        for(size_t i = 0; i < output.size(); i++) {
            mse += pow(output[i] - target[i], 2);
        }
        mse /= output.size();
        double rmse = sqrt(mse);
        
        for(size_t i = 0; i < output.size(); i++) {
            derivative[i] = (output[i] - target[i]) / (output.size() * rmse);
        }
    }
    return derivative;
}

vector<double> Activation(const vector<double>& train_data, const string& activate="SIGMOID") {
    vector<double> result(train_data.size());
    if (activate == "SOFTMAX"){
        double max = *max_element(train_data.begin(),train_data.end());
        double sum = 0.0;
        for (size_t i = 0; i < train_data.size(); i++)
        {
            result[i] = exp(train_data[i]-max);
            sum+=result[i];
        };
        for (size_t i = 0; i < train_data.size(); i++)
        {
           result[i] = result[i]/sum;
        } 
        return result;    
    }
    for(size_t i = 0; i < train_data.size(); i++) {
        if (activate == "SIGMOID") {
            result[i] = 1.0 / (1.0 + exp(-train_data[i]));
        }else if (activate == "RELU") {
            result[i] = max(0.0, train_data[i]);
        }else if (activate == "TANH") {
            result[i] = tanh(train_data[i]);
        }
    }    
    return result;
}
vector<double> ActivationDerivative(const vector<double>& z, const string& activate="SIGMOID") {
    vector<double> derivative(z.size());
    
    if (activate == "SOFTMAX") {
        // Softmax的导数与交叉熵损失一起使用时会简化
        return z;
    }
    for(size_t i = 0; i < z.size(); i++) {
        if (activate == "SIGMOID") {
            derivative[i] = z[i] * (1.0 - z[i]);
        }
        else if (activate == "RELU") {
            derivative[i] = z[i] > 0 ? 1.0 : 0.0;
        }
        else if (activate == "TANH") {
            derivative[i] = 1.0 - z[i] * z[i];
        }
    }
    return derivative;
}

struct Layer {
    vector<vector<double>> weights;
    vector<double> bias;
    vector<double> outputs;
    vector<double> deltas;
    vector<double> z;
    ActivationFunction activation;
    Layers layer;
    double dropout_rate;
    vector<bool> dropout_mask;
    bool is_train = true;

    Layer(const int input_size, const int output_size,
          ActivationFunction act_type, Layers layer_type) {
        weights.resize(output_size, vector<double>(input_size));
        bias.resize(output_size, 0.0);
        outputs.resize(output_size);
        deltas.resize(output_size);
        z.resize(output_size);
        activation = act_type;
        layer = layer_type;
        dropout_rate = 0.2;  
    
        if (layer_type == Layers::Dropout) {
            dropout_mask.resize(output_size, true);  
        }
 
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, std::sqrt(2.0/input_size));
        
        for (size_t i = 0; i < output_size; i++) {
            for (size_t j = 0; j < input_size; j++) {
                weights[i][j] = dist(gen);
            }
        }
    }

    vector<double> make_layer(const vector<double>& train_data) {
        // 检查维度
        if(weights[0].size() != train_data.size()) {
            cout << "Input size mismatch. Expected: " << weights[0].size() 
                 << ", Got: " << train_data.size() << endl;
            return vector<double>();
        }
        
        z = MatrixOps::matrix_vector_multiply(weights, train_data);
        for (size_t i = 0; i < z.size(); ++i) {
            z[i] += bias[i];
        }

        if (layer == Layers::Dropout && is_train) {
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<double> dist(0.0, 1.0);
            
            for (size_t i = 0; i < z.size(); ++i) {
                dropout_mask[i] = (dist(gen) > dropout_rate);
                if (!dropout_mask[i]) {
                    z[i] = 0.0;
                } else {
                    z[i] /= (1.0 - dropout_rate); 
                }
            }
        }
        string activate_type;
        switch(activation) {
            case ActivationFunction::SIGMOID: activate_type = "SIGMOID"; break;
            case ActivationFunction::RELU:   activate_type = "RELU";    break;
            case ActivationFunction::TANH:   activate_type = "TANH";    break;
            case ActivationFunction::SOFTMAX: activate_type = "SOFTMAX"; break;
        }

        outputs = Activation(z,activate_type);
        return outputs;
    }
};

class NeuralNetwork {
private:
    vector<Layer> layers;
    vector<int> topology;
    int batch_size;
    double learning_rate;
    vector<double> inputs;

public:
    NeuralNetwork(const vector<int> topology,int batch_size=16,double lr=0.01)
    : topology(topology),batch_size(batch_size),learning_rate(lr){
        if(topology.size() < 2) {
            cout << "Error: Network must have at least input and output layers" << endl;
            return;
        }
        
        cout << "Creating network with topology: ";
        for(int size : topology) {
            cout << size << " ";
        }
        cout << endl;
        
        for (size_t i = 0; i < topology.size()-1; i++) {
            layers.emplace_back(topology[i], topology[i+1], 
                              ActivationFunction::SIGMOID, Layers::Linear);
        }
    }

    vector<double> forward(const vector<double>& train_data){
        vector<double> current=train_data;
        inputs=train_data;
        for (auto& layer:layers)
        {
            current=layer.make_layer(current);
        }
        return current;

    }

    vector<double> backward(
    const vector<double>& outputs,
    const vector<double>& targets,
    SGD& Optimizer,const string& loss_function = "MSE") {
        vector<double> output_error = LossDerivative(outputs,targets);
        
        for (int i = layers.size()-1; i >= 0; i--) {
            Layer& current_layer = layers[i];
            string activate_type;
            switch(current_layer.activation) {
                case ActivationFunction::SIGMOID: activate_type = "SIGMOID"; break;
                case ActivationFunction::RELU:   activate_type = "RELU"; break;
                case ActivationFunction::TANH:   activate_type = "TANH"; break;
                case ActivationFunction::SOFTMAX: activate_type = "SOFTMAX"; break;
            }
            
            // 计算当前层的delta
            vector<double> activation_derivative = ActivationDerivative(current_layer.outputs, activate_type);
            for(size_t j = 0; j < current_layer.deltas.size(); j++) {
                current_layer.deltas[j] = output_error[j] * activation_derivative[j];
            }
            
            // 获取输入
            vector<double> layer_input = (i == 0) ? inputs : layers[i-1].outputs;
            
            // 初始化权重梯度矩阵
            vector<vector<double>> weight_gradients(current_layer.weights.size(), 
                vector<double>(current_layer.weights[0].size()));
            
            // 计算权重梯度
            for(size_t j = 0; j < current_layer.weights.size(); j++) {
                for(size_t k = 0; k < current_layer.weights[0].size(); k++) {
                    weight_gradients[j][k] = current_layer.deltas[j] * layer_input[k];
                }
            }
            
            // 更新权重和偏置
            Optimizer.update(current_layer.weights, current_layer.bias,
                            weight_gradients, current_layer.deltas);
            
            // 计算前一层的误差
            if(i > 0) {
                vector<double> prev_error(layer_input.size(), 0.0);
                for(size_t j = 0; j < current_layer.weights.size(); j++) {
                    for(size_t k = 0; k < current_layer.weights[0].size(); k++) {
                        prev_error[k] += current_layer.weights[j][k] * current_layer.deltas[j];
                    }
                }
                output_error = prev_error;
            }
        }
        return output_error;
    }
    
    void train(const vector<vector<double>>& train_data,  
          const vector<vector<double>>& targets,     
          int epochs,
          int batch_size){
        SGD Optimizer(learning_rate);
        int num_batches = (train_data.size() + batch_size -1)/batch_size;
        for (size_t epoch = 0; epoch < epochs; epoch++)
        {
            double epoch_loss = 0.0;
            for (size_t batch = 0; batch < num_batches; batch++)
            {
                double batch_loss = 0.0;
                int start_idx=batch*batch_size;
                int end_idx = min(start_idx + batch_size, (int)train_data.size());
                for (size_t t = start_idx; t < end_idx; t++)
                {
                    vector<double> outputs=forward(train_data[t]);
                    backward(outputs,targets[t],Optimizer);
                    vector<double> error;
                    error.resize(outputs.size());
                    for(size_t i = 0; i < outputs.size(); i++) {
                        error[i] = outputs[i] - targets[t][i];
                    }
    
                    batch_loss += Loss(error, "MSE");  
                }
                batch_loss /= (end_idx - start_idx);
                epoch_loss += batch_loss;
            }
            epoch_loss /= num_batches;
            
            // 每隔一定epoch打印损失
            if(epoch % 1 == 0) {
                cout << "Epoch " << epoch << ", Loss: " << epoch_loss << endl;
        }
        }
    }
};



int main() {
    vector<int> topology = {2, 8, 8, 1};
    // 减小学习率，增加训练轮数
    NeuralNetwork nn(topology, 4, 0.1);
    
    // 3. 准备XOR训练数据
    vector<vector<double>> train_data = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    
    vector<vector<double>> targets = {
        {0},
        {1},
        {1},
        {0}
    };
    
    // 4. 训练网络
    cout << "Training started..." << endl;
    nn.train(train_data, targets, 10000, 4);
    
    // 5. 测试网络
    cout << "\nTesting the network:" << endl;
    for(size_t i = 0; i < train_data.size(); i++) {
        vector<double> output = nn.forward(train_data[i]);
        cout << train_data[i][0] << " XOR " << train_data[i][1] 
             << " = " << output[0] << " (expected: " << targets[i][0] << ")" << endl;
    }
    
    return 0;
}