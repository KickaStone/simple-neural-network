//
// Created by JunchengJi on 7/31/2023.
//

#ifndef NETWORK_POOLING_H
#define NETWORK_POOLING_H

#include "layer.h"
#include "../utils/convolution.h"

enum class PoolingType{
    MAX,
    AVG,
};

class Pooling : public Layer{
private:
    int _inputChannel;
    int _inputHeight;
    int _inputWidth;
    int _kenrelSize;
    int _stride;
    int _padding;

    int _outputHeight;
    int _outputWidth;

    double *_output_v;

    std::vector<MatMap> _input_m;
    std::vector<Mat> _output_m;
    std::vector<Mat> _record;

    PoolingType poolingType;

    double *_input_grad_v;
    std::vector<Mat> _input_grad_m;

public:
    Pooling(int inputChannel, int inputHeight, int inputWidth, int kenrelSize, int stride, int padding, PoolingType type);
    ~Pooling() override;

    double* forward(const double *x) override;
    double* backward(const double *grad) override;
    void update(double lr, int batchSize) override;
};


#endif //NETWORK_POOLING_H
