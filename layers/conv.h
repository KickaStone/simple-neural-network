#ifndef CONV_H
#define CONV_H

#include <numeric>
#include <utility>
#include <fstream>

#include "layer.h"
#include "../utils/convolution.h"
#include <chrono>


using Mat3d = std::vector<Mat>;
using MatMap = Eigen::Map<Mat>;
class Conv : public Layer {
private:
    int _inputChannel; 
    int _inputHeight;
    int _inputWidth;
    int _outputChannel;
    int _kernel_size;
    int _stride;
    int _padding;
    int _outputHeight;
    int _outputWidth;

    std::vector<Mat3d> K;   // kernels (h * w * inputchannel * outputchannel)
    Vec b; // bias

    Mat3d _input;
    double* _output;
    double* _grad;
    Mat3d a;
    std::vector<Mat3d> dK;
    Vec db;

public:
    Conv(int inputChannel, int inputHeight, int inputWidth, int outputChannel, int kernel_size, int stride, int padding, Activation::ActivationFunctionType type);
    ~Conv() override;
    
    double* forward(const double *data) override;
    double* backward(const double *grad) override;
    void update(double lr, int batchSize) override;

    void setKernel(int i, double* kernel);
    void saveKernel(std::string path);
};
#endif // CONV_H
