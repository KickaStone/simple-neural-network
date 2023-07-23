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
            z[tid] += a[tid] * w[i * outputSize + tid];
        out[tid] = sigmoid(z[tid]);
    }
}

/**
 * @param input input vector size
 * @param output output vector size
 * @param w weight matrix
*/
__global__ void backprpoError(float *delta, float *w, float *deltaNext, float *z, int input, int ouput){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input)
    {
        float sum = 0.0f;
        for (int i = 0; i < ouput; i++)
            sum += deltaNext[i] * w[tid * ouput + i];
        delta[tid] = sum * sigmoidDerivative(z[tid]);
    }
    
}

/**
 * @param inputSize 
*/
__global__ void get_nabla_w(float *delta, float *z, float *nabla_w, int inputSize, int outputSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < inputSize * outputSize)
    {
        int i = tid / outputSize;
        int j = tid % outputSize;
        nabla_w[tid] = delta[j] * z[i];
        // printf("%d %d %d %f %f %f\n", tid, i, j, delta[j], z[i], nabla_w[tid]);
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
        printf("%f ", a[tid]);
}

__global__ void sigmoid_prime_vec(float *a, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        a[tid] = sigmoidDerivative(a[tid]);
    }
}