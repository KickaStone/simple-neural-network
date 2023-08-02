//
// Created by JunchengJi on 7/31/2023.
//

#include "pooling.h"

pooling::pooling(int volumeSize, int Nx, int Ny, int Nk, int p,
                 int stride, PoolingType type1, Activation::ActivationFunctionType type2) : Layer(Nx * Ny * volumeSize,
                ((Nx + 2 * p - Nk) / stride + 1) * ((Ny + 2 * p - Nk) / stride + 1) * volumeSize,
                type2)
{
    this->Nk = Nk;
    this->stride = stride;
    this->Nx = Nx;
    this->Ny = Ny;
    this->Ox = (Nx + 2 * p - Nk) / stride + 1;
    this->Oy = (Ny + 2 * p - Nk) / stride + 1;
    this->volumeSize = volumeSize;
    this->padding = p;
    this->poolingType = type1;

    this->output = new double[Ox * Oy * volumeSize];
    this->record = new int[Ox * Oy * volumeSize];

    for (int i = 0; i < Ox * Oy * volumeSize; ++i)
    {
        output[i] = 0;
        record[i] = 0;
    }
    input_grad = new double[Nx * Ny * volumeSize];
}

pooling::~pooling()
{
    delete[] output;
    delete[] record;
    delete[] input_grad;
}

double *pooling::forward(const double *x)
{
    this->input = x;
    switch (this->poolingType)
    {
    case PoolingType::AVG:
        for (int i = 0; i < volumeSize; i++)
        {
            convolution::avg_pooling(input + i * Nx * Ny, Nx, Ny, Nk, stride, output + i * Ox * Oy);
        }
        break;
    case PoolingType::MAX:
        for (int i = 0; i < volumeSize; i++)
        {
            convolution::max_pooling(input + i * Nx * Ny, Nx, Ny, Nk, stride, output + i * Ox * Oy, record + i * Ox * Oy);
        }
        break;
    default:
        throw "pooling type not supported!";
    }
    return output;
}

double *pooling::backward(const double *grad)
{
    double *output_grad;
    output_grad = new double[Ox * Oy * volumeSize];
    std::copy(grad, grad + Ox * Oy * volumeSize, output_grad);

    for (int i = 0; i < volumeSize; i++)
    {
        for (int j = 0; j < Ox * Oy; j++)
        {
            output_grad[i * Ox * Oy + j] *= derivative(output[i * Ox * Oy + j]);
        }
    }

    std::fill(input_grad, input_grad + Nx * Ny * volumeSize, 0);
    switch (poolingType)
    {
    case PoolingType::AVG:
        for (int v = 0; v < volumeSize; v++)
        {
            int idx = v * Nx * Ny;
            int Oidx = v * Ox * Oy;
            int i = 0, j = 0;
            int Oi = 0, Oj = 0;
            for (Oi = 0; Oi < Ox; Oi++)
            {
                for (Oj = 0; Oj < Oy; Oj++)
                {
                    //                        printf("Oi: %d, Oj: %d\n", Oi, Oj);
                    int index = Oi * Oy + Oj;
                    for (i = Oi * stride; i < Oi * stride + Nk; i++)
                    {
                        for (j = Oj * stride; j < Oj * stride + Nk; j++)
                        {
                            //                                printf("\ti: %d, j: %d, index: %d\n", i, j, idx + i * Ny + j);
                            input_grad[idx + i * Ny + j] += output_grad[Oidx + index] / (Nk * Nk);
                        }
                    }
                }
            }
        }
        break;
    case PoolingType::MAX:
        // based on the index of max value, set the gradient
        for (int v = 0; v < volumeSize; v++)
        {
            for (int i = 0; i < Ox; i++)
            {
                for (int j = 0; j < Oy; j++)
                {
                    int index = i * Oy + j;
                    input_grad[v * Nx * Ny + record[index]] += output_grad[v * Ox * Oy + index];
                }
            }
        }
        break;
    default:
        break;
    }
    return input_grad;
}

void pooling::update(double lr, int batchSize)
{
    // do nothing
}
