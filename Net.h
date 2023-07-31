//
// Created by JunchengJi on 7/30/2023.
//

#ifndef NETWORK_NET_H
#define NETWORK_NET_H

#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include "layers/layer.h"

class Net {
protected:
    int num_layers{};
    int output_dim{};
    std::vector<Layer*> layers;
public:
    Net() = default;
    ~Net() = default;
    virtual double* forward(double *input_data) = 0;
    virtual double* backward(double *grad) = 0;
    virtual void update(double lr, int batchSize) = 0;
};

#endif //NETWORK_NET_H
