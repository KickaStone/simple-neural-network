#pragma once
#include <stdio.h>

__global__ void vecAdd(float* a, float* b, float* c, int n);

__global__ void vecAdd2(float *a, float *b, int n);

__global__ void vecSub(float* a, float* b, float* c, int n);

__global__ void vecMul(float* a, float* b, float* c, int n);

__global__ void matVecMul(float* a, float* b, float* c, int n);

__global__ void matAdd(float* a, float* b, int n);

__device__ float sigmoid(float x);

__device__ float sigmoidDerivative(float x);

__global__ void forward(float *a, float *w, float *z, float *b, float *output, int inputSize, int outputSize);

__global__ void backprpoError(float *delta, float *w, float *deltaNext, float *z, int inputSize, int outputSize);

__global__ void get_nabla_w(float *delta, float *z, float *nabla_w, int inputSize, int outputSize);

__global__ void get_nabla_b(float *delta, float *nabla_b, int outputSize);

__global__ void vecScale(float *a, float scale, int n);

__global__ void print(float *a, int n);

__global__ void vecSub2(float *a, float *b, int n);


__global__ void sigmoid_prime_vec(float *a, int n);