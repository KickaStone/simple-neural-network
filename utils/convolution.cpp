//
// Created by JunchengJi on 7/31/2023.
//
#include "convolution.h"

void convolution::cross_correlation(const double *input, const double *kernel, int Nx, int Ny, int Nk, int stride, double *output){
    int Ox = (Nx - Nk) / stride + 1;
    int Oy = (Ny - Nk) / stride + 1;
    int ii, jj;
    for(int i = 0; i < Ox; i++){
        for(int j = 0; j < Oy; j++){
            if(i + Nk > Nx || j + Nk > Ny) continue; // out of bound
            double sum = 0;
            ii = i*stride;
            jj = j*stride;
            for(int k = 0; k < Nk; k++){
                for(int l = 0; l < Nk; l++){
                    if(ii + k >= Nx || jj + l >= Ny) continue;
                    sum += input[(ii + k) * Nx + (jj + l)] * kernel[k * Nk + l];
                }
            }
            output[i * Ox + j] = sum;
        }
    }
}

void convolution::correlation(const double *input, const double *kernel, int Nx, int Ny, int Nk, int stride, double *output){
    int Ox = (Nx - Nk) / stride + 1;
    int Oy = (Ny - Nk) / stride + 1;
    int ii, jj;
    for(int i = 0; i < Ox; ++i){
        for(int j = 0; j < Oy; ++j){
            ii = i*stride;
            jj = j*stride;
            if(ii + Nk > Nx || jj + Nk > Ny) continue; // out of bound
            double sum = 0;
            for(int k = 0; k < Nk; k++){
                for(int l = 0; l < Nk; l++){
                    sum += input[(ii + k) * Nx + (jj + l)] * kernel[(Nk - k - 1) * Nk + (Nk - l - 1)];
                }
            }
            output[i * Ox + j] = sum;
        }
    }
}

void convolution::padding(const double *input, int Nx, int p, double *output){
    int Nx_p = Nx + 2 * p;
    for(int i = 0; i < Nx_p; i++){
        for(int j = 0; j < Nx_p; j++){
            if(i < p || i >= Nx_p - p || j < p || j >= Nx_p - p){
                output[i * Nx_p + j] = 0;
            } else {
                output[i * Nx_p + j] = input[(i - p) * Nx + (j - p)];
            }
        }
    }
}

void convolution::avg_pooling(const double *input, int Nx, int Ny, int Nk, int stride, double *output) {
    double kernel_val = 1.0 / (Nk * Nk);
    auto *kernel = new double[Nk * Nk];
    std::fill(kernel, kernel + Nk * Nk, kernel_val);
    convolution::correlation(input, kernel, Nx, Ny, Nk, stride, output);
}

void convolution::max_pooling(const double *input, int Nx, int Ny, int Nk, int stride, double *output, int *record) {
    int Ox = (Nx - Nk) / stride + 1;
    int Oy = (Ny - Nk) / stride + 1;
    int ii, jj, max_idx = 0;
    for(int i = 0; i < Ox; i++){
        for(int j = 0; j < Oy; j++){
            if(i + Nk > Nx || j + Nk > Ny) continue; // out of bound
            double max_val = std::numeric_limits<double>::lowest();
            ii = i*stride;
            jj = j*stride;
            if(ii + Nk > Nx || jj + Nk > Ny) continue; // out of bound
            for(int k = 0; k < Nk; k++){
                for(int l = 0; l < Nk; l++){
                    if(input[(ii + k) * Nx + (jj + l)] > max_val){
                        max_val = input[(ii + k) * Nx + (jj + l)];
                        max_idx = (ii + k) * Nx + (jj + l);
                    }
                }
            }
            output[i * Ox + j] = max_val;
            record[i * Ox + j] = max_idx;
        }
    }
}
