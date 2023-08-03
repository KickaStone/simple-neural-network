#ifndef CONV_H
#define CONV_H

#include "layer.h"

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

    std::vector<std::vector<Mat>> K;   // kernels (h * w * inputchannel * outputchannel)
    Vec b; // bias

    std::vector<MatMap> _input;
    double* _output;
    double* _grad;
    std::vector<Mat> a;
    std::vector<std::vector<Mat>> dK;
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
