//
// Created by JunchengJi on 7/31/2023.
//

#include "Pooling.h"

Pooling::Pooling(int volumeSize, int Nx, int Ny, int Nk, int p,
                 int stride, PoolingType type1, Activation::ActivationFunctionType type2) :
        Layer(
                                 Nx*Ny*volumeSize,
                                 ((Nx + 2*p - Nk) / stride + 1) * ((Ny + 2*p - Nk) / stride + 1),
                                 type2
                         ), Nx(Nx), Ny(Ny), Nk(Nk), stride(stride) {
    this->padding = p;
    Ox = (Nx + 2*padding - Nk) / stride + 1;
    Oy = (Ny + 2*padding - Nk) / stride + 1;
    this->volumeSize = volumeSize;
    this->poolingType = type1;

    this->output = new double[Ox * Oy * volumeSize];
    this->record = new int[Ox * Oy * volumeSize];

    for (int i = 0; i < Ox * Oy * volumeSize; ++i) {
        output[i] = 0;
        record[i] = 0;
    }
    input_grad = new double[Nx * Ny * volumeSize];
}

Pooling::~Pooling() {
    delete[] output;
    delete[] record;
    delete[] input_grad;
}

double *Pooling::forward(double *x) {
    this->input = x;
    switch(this->poolingType){
        case PoolingType::AVG:
            for(int i = 0; i < volumeSize; i++){
                convolution::avg_pooling(input + i * Nx * Ny, Nx, Ny, Nk, stride, output + i * Ox * Oy);
            }
            break;
        case PoolingType::MAX:
            for(int i = 0; i < volumeSize; i++){
                convolution::max_pooling(input + i * Nx * Ny, Nx, Ny, Nk, stride, output + i * Ox * Oy, record + i * Ox * Oy);
            }
            break;
        default:
            throw "Pooling type not supported!";
    }
    return output;
}

double *Pooling::backward(double *grad) {
    for(int i = 0; i < volumeSize; i++){
        for(int j = 0; j < Ox * Oy; j++){
            grad[i * Ox * Oy + j] *= derivative(output[i * Ox * Oy + j]);
        }
    }

    std::fill(input_grad, input_grad + Nx * Ny * volumeSize, 0);
    switch(poolingType){
        case PoolingType::AVG:
            for(int v = 0; v < volumeSize; v++){
                int idx = v * Nx * Ny;
                int Oidx = v * Ox * Oy;
                int i = 0, j = 0;
                int Oi = 0, Oj = 0;
                for(Oi = 0; Oi < Ox; Oi++){
                    for(Oj = 0; Oj < Oy; Oj++){
//                        printf("Oi: %d, Oj: %d\n", Oi, Oj);
                        int index = Oi * Oy + Oj;
                        for(i = Oi * stride; i < Oi * stride + Nk; i++){
                            for(j = Oj * stride; j < Oj * stride + Nk; j++){
//                                printf("\ti: %d, j: %d, index: %d\n", i, j, idx + i * Ny + j);
                                input_grad[idx + i * Ny + j] += grad[Oidx + index] / (Nk * Nk);
                            }
                        }
                    }
                }
            }
            break;
        case PoolingType::MAX:
            // based on the index of max value, set the gradient
            for(int v = 0; v < volumeSize; v++){
                for(int i = 0; i < Ox; i++){
                    for(int j = 0; j < Oy; j++){
                        int index = i * Oy + j;
                        input_grad[v * Nx * Ny + record[index]] += grad[v * Ox * Oy + index];
                    }
                }
            }
            break;
        default:
            break;
    }
    return input_grad;
}

void Pooling::update(double lr, int batchSize) {
//do nothing
}



