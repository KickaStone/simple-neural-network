//
// Created by JunchengJi on 7/30/2023.
//

#ifndef NETWORK_ACTIVATION_H
#define NETWORK_ACTIVATION_H

#include <string>
#include <functional>
#include <algorithm>
#include <cmath>

namespace Activation{
    inline double sigmoid(double x){
        return 1.0 / (1.0 + exp(-x));
    }

    inline double sigmoid_derivative(double x){
        return x * (1.0 - x);
    }

    inline double relu(double x){
        return std::max(0.0, x);
    }

    inline double relu_derivative(double x){
        return x > 0.0 ? 1.0 : 0.0;
    }

    inline double tanh(double x){
        return std::tanh(x);
    }

    inline double tanh_derivative(double x){
        return 1.0 - x * x;
    }

    enum class ActivationFunctionType{
        RELU,
        SIGMOID,
        TANH,
        NONE
    };

    struct ActivationFunctions{
        std::function<double(double)> activation;
        std::function<double(double)> derivative;
        explicit ActivationFunctions(ActivationFunctionType func){
            switch(func){
                case ActivationFunctionType::RELU:
                    activation = relu;
                    derivative = relu_derivative;
                    break;
                case ActivationFunctionType::SIGMOID:
                    activation = sigmoid;
                    derivative = sigmoid_derivative;
                    break;
                case ActivationFunctionType::TANH:
                    activation = tanh;
                    derivative = tanh_derivative;
                    break;
                case ActivationFunctionType::NONE:
                    activation = [](double x){return x;};
                    derivative = [](double x){return 1.0;};
                    break;
                default:
                    throw "Activation function not supported";
            }
        }
    };
}

#endif //NETWORK_ACTIVATION_H


