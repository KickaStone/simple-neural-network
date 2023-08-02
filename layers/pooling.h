//
// Created by JunchengJi on 7/31/2023.
//

#ifndef NETWORK_POOLING_H
#define NETWORK_POOLING_H

#include "layer.h"
#include "../utils/convolution.h"

enum class PoolingType{
    MAX_POOL_2D,
    AVG_POOL_2D,
};

class pooling : public Layer{
private:
    int _inputChannel;
    int _inputHeight;
    int _inputWidth;
    int _kenrelSize;
    int _stride;
    int _padding;

    int _outputHeight;
    int _outputWidth;

    double *_input_v;
    double *_output_v;

    std::vector<Mat> _input_m;
    std::vector<Mat> _output_m;
    std::vector<Mat> _record;

    PoolingType poolingType;

    double *_grad_v;
    std::vector<Mat> _grad_m;

public:
    pooling(int inputChannel, int inputHeight, int inputWidth, int kenrelSize, int stride, int padding, PoolingType type);
    ~pooling() override;

    double* forward(const double *x) override;
    double* backward(const double *grad) override;
    void update(double lr, int batchSize) override;
};


#endif //NETWORK_POOLING_H
