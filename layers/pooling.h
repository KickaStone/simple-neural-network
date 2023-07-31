//
// Created by JunchengJi on 7/31/2023.
//

#ifndef NETWORK_POOLING_H
#define NETWORK_POOLING_H

#include "layer.h"
#include "../utils/convolution.h"

enum class PoolingType{
    MAX,
    AVG
};

class pooling : public Layer{
private:
    int Nk;
    int stride;
    int Nx, Ny;
    int Ox, Oy;
    int volumeSize;
    int padding;
    PoolingType poolingType;

    // input : Nx * Ny * volumeSize
    const double* input;
    double* output;
    double* input_grad;
    int* record; // record the index of max value

public:
    pooling(int volumeSize, int Nx, int Ny, int Nk, int p, int stride, PoolingType type1, Activation::ActivationFunctionType type2);
    ~pooling() override;

    double* forward(const double *x) override;
    double* backward(const double *grad) override;
    void update(double lr, int batchSize) override;
//    void setInput(double *input);
//    double* getZ();
};


#endif //NETWORK_POOLING_H
