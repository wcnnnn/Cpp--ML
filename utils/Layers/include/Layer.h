#pragma once
#include <vector>
#include <string>
#include "../../Common/include/Activation.h"
#include "../../Common/include/Optimizer.h"

class Layer {
protected:
    std::string layer_name;
    bool is_training;
    ActivationFunction activation;

public:
    Layer(const std::string& name, ActivationFunction act = ActivationFunction::RELU)
        : layer_name(name), is_training(true), activation(act) {}
    
    virtual ~Layer() = default;

    // 基础接口
    virtual void initialize() = 0;
    virtual void setTrainingMode(bool is_train) { is_training = is_train; }
    
    // Getters
    virtual std::string getName() const { return layer_name; }
    virtual bool isTraining() const { return is_training; }
};