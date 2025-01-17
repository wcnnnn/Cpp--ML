#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <string>
using namespace std;

class NaiveBayes
{
private:
    map<int,double> prior_probabilities;
    map<int,vector<pair<double,double>>> continuous_params;
    map<int,map<int,map<double,double>>> discrete_params;
    vector<bool> is_continuous;
    int feature_dimension;
    int class_num;
    
public:
    NaiveBayes(vector<bool> is_continuous);
    ~NaiveBayes();
    void train(vector<vector<double>> train_data,vector<double> target_label);
};

NaiveBayes::NaiveBayes(vector<bool> is_continuous)
{
    if(is_continuous.empty()){
        cout<<"is_continuous error"<<endl;
        return;
    }
    this->is_continuous = is_continuous;
    this->feature_dimension = is_continuous.size();
    this->class_num = 0;
}

void NaiveBayes::train(vector<vector<double>> train_data,vector<double> target_label)
{
    if(train_data.empty() or target_label.empty()){
        cout<<"train_data or target_label error"<<endl;
        return;
    }
    if(train_data[0].size()!=feature_dimension){
        cout<<"train_data error"<<endl;
        return;
    }
    map<int,int> class_count;
    for(int i=0;i<target_label.size();i++){
        class_count[target_label[i]]++;
    }
    for(int i=0;i<class_count.size();i++){
        prior_probabilities[i]=class_count[i]/target_label.size();
    }
    map<int,map<int,map<double,double>>> statis;
    for(const auto& class_pair : class_count){
        int class_label = class_pair.first;
        int class_count = class_pair.second;
        vector<double> mean(feature_dimension,0.0);
        vector<double> variance(feature_dimension,0.0);
        for(int i=0;i<train_data.size();i++){
            for(int j=0;j<feature_dimension;j++){
                if(is_continuous[j]){
                    mean[j]+=train_data[i][j];
                }
                else{

                }

            }
        }
             
    }
        

}



NaiveBayes::~NaiveBayes()
{
}
