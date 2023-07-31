#ifndef CONV_H
#define CONV_H

#include <numeric>
#include <utility>

#include "layer.h"
#include "../utils/convolution.h"

class Conv : public Layer {
private:
    std::vector<double*> kernel;
    double* b;
    const double* input;                    // TODO: volumes support
    double* a;
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
    Conv(int Nx, int Ny, int kernel_size,
         int stride, int n_kernel, int padding, Activation::ActivationFunctionType type);
    ~Conv() override;
    
    double* forward(const double *data) override;
    double* backward(const double *grad) override;
    void update(double lr, int batchSize) override;

    void setKernel(int i, double *myKernel);
    void setBias(int i, double bias);
    void setInput(double *x);

    std::vector<double*>& getDk();
};
#endif // CONV_H
