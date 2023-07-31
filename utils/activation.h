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
}

#endif //NETWORK_ACTIVATION_H


