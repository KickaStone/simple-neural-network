//
// Created by JunchengJi on 7/31/2023.
//

#ifndef NETWORK_CONVOLUTION_H
#define NETWORK_CONVOLUTION_H

#include <numeric>

namespace convolution{
    void padding(const double *input, int Nx, int p, double *output);
    void correlation(const double *input, const double *kernel, int Nx, int Ny, int Nk, int stride, double *output);
    void cross_correlation(const double *input, const double *kernel, int Nx, int Ny, int Nk, int stride, double *output);
    void max_pooling(const double *input, int Nx, int Ny, int Nk, int stride, double *output, int *record);
    void avg_pooling(const double *input, int Nx, int Ny, int Nk, int stride, double *output);
}

#endif //NETWORK_CONVOLUTION_H
