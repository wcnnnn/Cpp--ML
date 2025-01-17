#include "iostream"
#include "vector"
#include "cmath"
#include <algorithm> 
#include <map>
#include <limits> 
using namespace std;


struct KDTree {
    vector<double> data;
    int label;
    int split_dimension;
    KDTree* left;
    KDTree* right;


    KDTree(const vector<double>& point, const int& l) :
        data(point), label(l), split_dimension(0), left(nullptr), right(nullptr) {}
    
    double compute_distance(const vector<double>& a, const vector<double>& b) {
        double dist = 0.0;
        for(size_t i = 0; i < a.size(); i++) {
            dist += pow(a[i] - b[i], 2);
        }
        return sqrt(dist);
    }

    KDTree* build_kdtree(vector<vector<double>> data, vector<int> label, int depth) {
        if(data.empty()) return nullptr;
        if(data.size() == 1) return new KDTree(data[0], label[0]);

        int split_dimension = depth % data[0].size();
        vector<pair<double,int>> split_data;
        for(int i = 0; i < data.size(); i++) {
            split_data.push_back(make_pair(data[i][split_dimension], i));
        }
        sort(split_data.begin(), split_data.end());

        int mid = data.size() / 2;
        vector<vector<double>> left_data;
        vector<int> left_label;
        vector<vector<double>> right_data;
        vector<int> right_label;

        for(int i = 0; i < data.size(); i++) {
            if(i < mid) {
                left_data.push_back(data[split_data[i].second]);
                left_label.push_back(label[split_data[i].second]);
            } else {
                right_data.push_back(data[split_data[i].second]);
                right_label.push_back(label[split_data[i].second]);
            }
        }

        KDTree* node = new KDTree(data[split_data[mid].second], label[split_data[mid].second]);
        node->split_dimension = split_dimension;
        node->left = build_kdtree(left_data, left_label, depth + 1);
        node->right = build_kdtree(right_data, right_label, depth + 1);
        return node;
    }

    void search_knn(const vector<double>& point, int k, 
                   vector<pair<double,int>>& result, int depth = 0) {
        if(this == nullptr) return;

        double dist = compute_distance(point, data);
        result.push_back(make_pair(dist, label));

        int dim = depth % data.size();
        bool go_left = point[dim] < data[dim];

        if(go_left) {
            if(left != nullptr) left->search_knn(point, k, result, depth + 1);
            if(right != nullptr && (result.size() < k || 
               abs(point[dim] - data[dim]) < result[k-1].first)) {
                right->search_knn(point, k, result, depth + 1);
            }
        } else {
            if(right != nullptr) right->search_knn(point, k, result, depth + 1);
            if(left != nullptr && (result.size() < k || 
               abs(point[dim] - data[dim]) < result[k-1].first)) {
                left->search_knn(point, k, result, depth + 1);
            }
        }

        sort(result.begin(), result.end());
        if(result.size() > k) {
            result.resize(k);
        }
    }
};


class KNN
{
private:
    int k;
    int dimension;
    vector<vector<double>> train_data;
    vector<int> target_label;
    KDTree* kdtree;
public:
    KNN(int initial_k,
    int initial_dimension,
    vector<vector<double>> initial_train_data,
    vector<int> initial_target_label)
    {
        if(initial_k<=0){
            cout<<"k error"<<endl;
            return;
        }
        if(initial_dimension<=0){
            cout<<"dimension error"<<endl;
            return;
        }
        if(initial_train_data[0].size()!=initial_dimension){
            cout<<"train_data error"<<endl;
            return;
        }
        if(initial_train_data.size()!=initial_target_label.size()){
            cout<<"target_label error"<<endl;
            return;
        }
        k=initial_k;
        dimension=initial_dimension;
        train_data=initial_train_data;
        target_label=initial_target_label;
        kdtree = (new KDTree(train_data[0], target_label[0]))->build_kdtree(train_data, target_label, 0);
    }

    int predict(const vector<double> test_data){
        if(test_data.size()!=dimension){
            cout<<"test_data error"<<endl;
            return 0;
        }
        vector<pair<double,int>> knn_result;
        kdtree->search_knn(test_data,k,knn_result);
        map<int, int> label_count;
        for(const auto&pair : knn_result){
            label_count[pair.second]++;
        }
        int max_count=0;
        int predict_label=0;
        for(const auto& pair : label_count){
            if(pair.second>max_count){
                max_count=pair.second;
                predict_label=pair.first;
            }
        }
        return predict_label;
    }

    vector<int> predict_batch(vector<vector<double>> test_batch){
        vector<int> predict_labels;
        for(int i=0; i<test_batch.size();i++){
            predict_labels.push_back(predict(test_batch[i]));
        }
        return predict_labels;
    }

    double accuracy(
        const vector<vector<double>> test_batch,
        const vector<int> test_label){
        double correct=0;
        for(int i=0;i<test_batch.size();i++){
            if(predict(test_batch[i]) == test_label[i]){
                correct++;
            }
        }
        return correct/double(test_batch.size());
    }

    vector<vector<double>> standardize(const vector<vector<double>> data,
    string method="minmax"){
        if(method=="minmax"){
            vector<double> min_values(dimension,numeric_limits<double>::max());
            vector<double> max_values(dimension,numeric_limits<double>::min());
            for(int i=0;i<data.size();i++){
                for(int j=0;j<dimension;j++){
                    min_values[j]=min(min_values[j],data[i][j]);
                    max_values[j]=max(max_values[j],data[i][j]);
                }
            }
            vector<vector<double>> standardized_data(data.size(),vector<double>(dimension));
            for(int i=0;i<data.size();i++){
                for(int j=0;j<dimension;j++){
                    standardized_data[i][j]=(data[i][j]-min_values[j])/(max_values[j]-min_values[j]);
                }
            }
            return standardized_data;
        }
        else if(method=="zscore"){
            vector<double> mean(dimension,0);
            vector<double> std(dimension,0);
            for(int i=0;i<data.size();i++){
                for(int j=0;j<dimension;j++){
                    mean[j]+=data[i][j];
                }
            }
            for(int j=0;j<dimension;j++){
                mean[j] /= data.size();
            }
            for(int i=0;i<data.size();i++){
                for(int j=0;j<dimension;j++){
                    std[j]+=pow(data[i][j]-mean[j],2);
                }   
            }
            for(int j=0;j<dimension;j++){
                std[j] = sqrt(std[j]/data.size());
            }
            vector<vector<double>> standardized_data(data.size(), vector<double>(dimension)); 
            for(int i=0;i<data.size();i++){
                for(int j=0;j<dimension;j++){
                    standardized_data[i][j]=(data[i][j]-mean[j])/std[j];
                }
            }
            return standardized_data;
        }
        else{
            cout<<"none support method"<<endl;
            return data;
        }
    }

    double compute_distance(vector<double> x,vector<double> y,
    string type = "Euclidean"){
        if(x.size()!=y.size()){
            cout<<"x and y size error"<<endl;
            return 0;
        }

        double sum=0;
        if(type =="Euclidean"){
            for(int i=0;i<x.size();i++){
                sum+=pow(x[i]-y[i],2);
            }
            return sqrt(sum);
        }
        else if(type =="Manhattan"){
            for(int i=0;i<x.size();i++){
                sum+=fabs(x[i]-y[i]);
            }
            return sum;
        }
        else if(type =="Chebyshev"){
            double max_distance=0;
            for(int i=0;i<x.size();i++){
                max_distance=max(max_distance,fabs(x[i]-y[i]));
            }
            return max_distance;
        }
        else{
            cout<<"none support type"<<endl;
            return 0;
        }
    }
 
};

int main() {
    // 1. 准备训练数据
    vector<vector<double>> train_data = {
        // 类别1的样本 (集中在原点附近)
        {1.0, 1.0}, {1.5, 2.0}, {2.0, 1.0}, {2.0, 2.0}, {1.0, 2.0},
        // 类别2的样本 (集中在右上方)
        {8.0, 8.0}, {8.5, 9.0}, {9.0, 8.0}, {8.0, 9.0}, {9.0, 9.0},
        // 类别3的样本 (集中在右下方)
        {8.0, 1.0}, {9.0, 2.0}, {8.0, 2.0}, {9.0, 1.0}, {8.5, 1.5}
    };
    
    vector<int> target_label = {
        1, 1, 1, 1, 1,          // 5个类别1
        2, 2, 2, 2, 2,          // 5个类别2
        3, 3, 3, 3, 3           // 5个类别3
    };
    
    // 2. 创建KNN分类器 (k=3, 2维数据)
    KNN knn(3, 2, train_data, target_label);
    
    // 3. 准备测试数据
    vector<vector<double>> test_data = {
        {1.2, 1.4},   // 应该预测为类别1
        {8.2, 8.5},   // 应该预测为类别2
        {8.5, 1.8},   // 应该预测为类别3
        {2.0, 8.0},   // 这是一个不确定的点
        {5.0, 5.0}    // 这是一个中间点
    };
    
    vector<int> test_label = {1, 2, 3, 2, 2};  // 真实标签
    
    // 4. 数据标准化
    cout << "使用Z-score标准化数据..." << endl;
    vector<vector<double>> standardized_train = knn.standardize(train_data, "zscore");
    vector<vector<double>> standardized_test = knn.standardize(test_data, "zscore");
    
    // 5. 创建新的KNN分类器（使用标准化后的数据）
    KNN standardized_knn(3, 2, standardized_train, target_label);
    
    // 6. 预测并输出结果
    cout << "\n预测结果：" << endl;
    vector<int> predictions = standardized_knn.predict_batch(standardized_test);
    
    for(size_t i = 0; i < test_data.size(); i++) {
        cout << "测试样本 " << i << ": (";
        for(size_t j = 0; j < test_data[i].size(); j++) {
            cout << test_data[i][j] << " ";
        }
        cout << ") -> 预测类别: " << predictions[i] 
             << ", 真实类别: " << test_label[i] << endl;
    }
    
    // 7. 计算并输出准确率
    double acc = standardized_knn.accuracy(standardized_test, test_label);
    cout << "\n准确率: " << acc * 100 << "%" << endl;
    
    return 0;
}