#pragma once
#include <cuda_runtime.h>
#include "common.h"

// __global__ void vecAdd(d_vec *a, d_vec *b, d_vec *c);

__global__ void memset(float *a, int n, float val);

__global__ void vecAdd(float *a, float *b, float *c, int n);

__global__ void matMulvec(float *a, float *b, float *c, int row, int col, bool transpose = false);

__global__ void sigmoid_ztoa(float *z, float *a, int n);

__global__ void sigmoid_z_prime(float *a, float *z_prime, int n);

__global__ void cost_prime(float *a, float *y, float *da, int n);

__global__ void vecMul(float *a, float *b, float *c, int n);

__global__ void copy(float *dst, float *src, int n);

__global__ void update(float *w, float *dC_dw, float eta, int n);

__global__ void cal_dw(float *a, float *delta, float *dC_dw, int input_size, int output_size);

__global__ void cal_loss(float *a, float *y, float *loss, int n);

