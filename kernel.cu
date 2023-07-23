#include "kernel.cuh"

__global__ void vecAdd(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

__global__ void vecAdd2(float *a, float *b, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        a[tid] += b[tid];
}

__global__ void vecSub(float *a, float *b, float*c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] - b[tid];
}

__global__ void vecSub2(float *a, float *b, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
        a[tid] -= b[tid];
}

__global__ void vecMul(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] * b[tid];
}

__global__ void matVecMul(float *a, float *b, float *c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += a[tid * n + i] * b[i];
        c[tid] = sum;
    }
}

__global__ void matAdd(float *a, float *b, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        a[tid] += b[tid];
}

__device__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoidDerivative(float x)
{
    return x * (1.0f - x);
}

__global__ void forward(float *a, float *w, float *z, float *b, float* out, int inputSize, int outputSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < outputSize)
    {
        z[tid] = b[tid];
        for (int i = 0; i < inputSize; i++)
            z[tid] += a[i] * w[tid * inputSize + i];
        out[tid] = sigmoid(z[tid]);
    }
}

/**
 * @param input input vector size
 * @param output output vector size
 * @param w weight matrix
*/
__global__ void backprpoError(float *delta, float *w, float *deltaNext, float *a, int input, int output){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < input){
        float sum = 0.0f;
        for(int i = 0; i < output; i++){
            sum += w[i * input + tid] * deltaNext[i];
        }
        delta[tid] = sum * sigmoidDerivative(a[tid]);
    }
}

/**
 * @param inputSize 
*/
__global__ void get_nabla_w(float *delta, float *z, float *nabla_w, int inputSize, int outputSize)
{
    int j = blockIdx.x;
    int k = threadIdx.x;
    if(j < outputSize && k < inputSize){
        nabla_w[j * inputSize + k] = delta[j] * z[k];
    }
}

__global__ void get_nabla_b(float *delta, float *nabla_b, int outputSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < outputSize){
        nabla_b[tid] = delta[tid];
    }
}

__global__ void vecScale(float *a, float scale, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        a[tid] *= scale;
}

__global__ void print(float *a, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size)
        printf("%.8lf ", a[tid]);
    printf("\n");
}

__global__ void sigmoid_prime_vec(float *z, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        z[tid] = sigmoid(z[tid]) * (1 - sigmoid(z[tid]));
    }
}
__global__ void cost_derivative(float *a, int label, float *delta, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        if(tid == label) delta[tid] = a[tid] - 1.0f;
        else delta[tid] = a[tid];
    }
}