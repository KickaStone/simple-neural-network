#ifndef CONV_H
#define CONV_H

#include "layer.h"

#include <numeric>

class Conv : public Layer {
private:
    std::vector<double*> kernel;
    double* b;
    double* input;                    // TODO: volumes support
    double* output;
    std::vector<double*> feature_map; // TODO: change to double*
    std::vector<double*> dk;          // TODO: change to double*
    double* db;
    double* dz;

    int n_kernel;
    int kernel_size;
    int stride;
    int p; // padding
    int Nx, Ny;
    int Ox, Oy;

public:
    static void padding(double *input, int Nx, int p, double *output);
    static void correlation(double *img, double *kernel, int Nx, int Nk, int stride, double *output);
    static void cross_correlation(double *img, double *kernel, int Nx, int Nk, int stride, double *output);

    Conv(int inputSize,Activation::ActivationFunctionType type, int Nx, int Ny, int kernel_size,
         int stride, int n_kernel, int padding);
    ~Conv();
    
    double* forward(double *input) override;
    double* backward(double *grad) override;
    void update(double lr, int batchSize) override;

    void setKernel(int i, double *kernel);
    void setBias(int i, double bias);
    void setInput(double *input);

    std::vector<double*>& getDk();

    
};
#endif // CONV_H
