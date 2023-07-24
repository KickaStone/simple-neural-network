#pragma once
#include <cuda_runtime.h>
#include "common.h"


__global__ void memset(double *a, int n, double val);

__global__ void vecAdd(double *a, double *b, double *c, int n);

__global__ void matMulvec(double *a, double *b, double *c, int row, int col, bool transpose = false);

__global__ void sigmoid_ztoa(double *z, double *a, int n);

__global__ void sigmoid_z_prime(double *a, double *z_prime, int n);

__global__ void cost_prime(double *a, double *y, double *da, int n);

__global__ void vecMul(double *a, double *b, double *c, int n);

__global__ void copy(double *dst, double *src, int n);

__global__ void update(double *w, double *dC_dw, double eta, int n);

__global__ void cal_dw(double *a, double *delta, double *dC_dw, int input_size, int output_size);

__global__ void cal_loss(double *a, double *y, double *loss, int n);