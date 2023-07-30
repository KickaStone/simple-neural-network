#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H

#include <utility>
#include <random>

#include "activation.h"


class Layer{
    public:
    Layer(int inputSize, int outputSize, Activation::ActivationFunctions af): inputSize(inputSize), outputSize(outputSize),
                                                                              activationFunc(std::move(af)){}
    virtual ~Layer()= default;
    virtual double* forward(double *input) = 0; // pure virtual function
    virtual double* backward(double *grad) = 0; // pure virtual function
    virtual void update(double lr, int batchSize) = 0; // pure virtual function
    [[nodiscard]] int getOutputSize() const{ return outputSize; }
    [[nodiscard]] int getInputSize() const{ return inputSize; }
protected:
    int inputSize;
    int outputSize;
    Activation::ActivationFunctions activationFunc;
};

#endif //NETWORK_LAYER_H