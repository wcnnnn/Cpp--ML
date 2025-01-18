#include <iostream>
#include <array>
#include <vector>
using namespace std;

class Perceptron{
private:
    vector<double> weights;
    double threshold;
    double learning_rate;
    double bias;
    int max_epoch;

public:
    Perceptron(
    int initial_weight_size,
    double initial_threshold=0.0, 
    double initial_bias = 0.0,
    double initial_learning_rate=0.01,
    int initial_max_epoch=100)
    {
        weights = vector<double> (initial_weight_size,0.5);
        threshold = initial_threshold;
        bias = initial_bias;
        learning_rate = initial_learning_rate;
        max_epoch = initial_max_epoch;
    }

    int predict(const vector<double>& inputs) {
        if(inputs.size() != weights.size()) {
            return -2;
        }
        double weight_sum = bias;
        for(size_t i = 0; i < inputs.size(); i++) {
            weight_sum += inputs[i] * weights[i];
        }
        return (weight_sum < threshold) ? -1 : 1;
    }
    void train(
    const vector<vector<double>>& inputs,
    const vector<int>& targets)
    {
        if(inputs.size() != targets.size()){
            cout << "input and target size mismatch" << endl;
            return;
        }
        cout << "start training" << endl;
        for(size_t i=0;i<weights.size();i++){
        cout << "initial weight:"<<i<<":"<<weights[i]<< endl;
        }
        cout << "initial bias:" << bias << endl;
        bool all_correct = false;
        int epoch = 0;
        while (!all_correct && epoch < max_epoch){
            all_correct = true;
            epoch++;
            cout << "epoch:" << epoch <<endl;
            for (size_t i =0; i< inputs.size(); i++){
                int result = predict(inputs[i]);
                if(result != targets[i]){
                    all_correct = false;
                    for(size_t j=0;j<weights.size();j++){
                        weights[j] = weights[j] + learning_rate * (targets[i] - result) * inputs[i][j];
                    }
                    bias = bias + learning_rate * (targets[i] - result);
                    for(size_t j=0;j<weights.size();j++){
                    cout << "new weight:"<<j<<":"<<weights[j]<< endl;
                    }
                    cout << "--------------------------------" << endl;
                }
            }

            if(all_correct){
                cout << "收敛" << endl;
            }
        }
        
        cout << "epoch:" << epoch << endl;
        for(size_t i=0;i<weights.size();i++){
        cout << "final weight:"<<i<<":"<<weights[i]<< endl;
        }
        cout << "final bias:" << bias << endl;
    }
};

int main() {
     Perceptron p(2,     
                0.0,    
                0.0,   
                0.1,    
                100);  

    vector<vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    vector<int> y = {-1, -1, -1, 1};

    cout << "训练数据：" << endl;
    for(size_t i = 0; i < X.size(); i++) {
        cout << "X[" << i << "]: " << X[i][0] << "," << X[i][1] 
             << " -> y: " << y[i] << endl;
    }

    p.train(X, y);

    cout << "\n测试结果：" << endl;
    vector<vector<double>> test_cases = {{0,0}, {0,1}, {1,0}, {1,1}};
    for(const auto& test : test_cases) {
        cout << "输入: " << test[0] << "," << test[1] 
             << " 预测: " << p.predict(test) << endl;
    }

    return 0;
}