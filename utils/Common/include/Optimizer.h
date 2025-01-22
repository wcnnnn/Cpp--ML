#pragma once
#include <vector>

enum class OptimizerType {
    SGD
};

class Optimizer {
protected:
    double learningRate;
    double momentum;
    double weightDecay;

public:
    Optimizer(double lr = 0.01, double momentum = 0.9, double weightDecay = 0.0001)
        : learningRate(lr), momentum(momentum), weightDecay(weightDecay) {}
    
    virtual ~Optimizer() = default;

    // 2D参数更新
    virtual void update(
        std::vector<std::vector<double>>& weights,
        std::vector<double>& bias,
        const std::vector<std::vector<double>>& weightGradients,
        const std::vector<double>& biasGradients
    ) = 0;

    // 3D参数更新
    virtual void update3D(
        std::vector<std::vector<std::vector<double>>>& weights,
        std::vector<double>& bias,
        const std::vector<std::vector<std::vector<double>>>& weightGradients,
        const std::vector<double>& biasGradients
    ) = 0;

    // Getters and setters
    double getLearningRate() const { return learningRate; }
    void setLearningRate(double lr) { learningRate = lr; }
    
    double getMomentum() const { return momentum; }
    void setMomentum(double m) { momentum = m; }
    
    double getWeightDecay() const { return weightDecay; }
    void setWeightDecay(double wd) { weightDecay = wd; }
};

class SGD : public Optimizer {
public:
    SGD(double lr = 0.01) : Optimizer(lr) {}
    
    void update(
        std::vector<std::vector<double>>& weights,
        std::vector<double>& bias,
        const std::vector<std::vector<double>>& weightGradients,
        const std::vector<double>& biasGradients
    ) override;

    void update3D(
        std::vector<std::vector<std::vector<double>>>& weights,
        std::vector<double>& bias,
        const std::vector<std::vector<std::vector<double>>>& weightGradients,
        const std::vector<double>& biasGradients
    ) override;
};