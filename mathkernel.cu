#include "mathkernel.cuh"

__global__ void memset(float *a, int n, float val)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        a[tid] = val;
    }
}

__global__ void vecAdd(float *a, float *b, float *c, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

__global__ void matMulvec(float *a, float *b, float *c, int row, int col, bool transpose)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float sum = 0;
    if(transpose){
        if(i < col){
            for(int j = 0; j < row; j++){
                sum += a[j * col + i] * b[j];
            }
            c[i] = sum;
        }
    }else{
        if(i < row){
            for(int j = 0; j < col; j++){
                sum += a[i * col + j] * b[j];
            }
            c[i] = sum;
        }
    }
}

__global__ void sigmoid_ztoa(float *z, float *a, int n)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        a[tid] = 1.0 / (1.0 + exp(-z[tid]));
    }
}

__global__ void sigmoid_z_prime(float *a, float *z_prime, int n)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        z_prime[tid] = a[tid] * (1 - a[tid]);
    }
}

__global__ void cost_prime(float *a, float *y, float *da, int n)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        da[tid] = a[tid] - y[tid];
    }
}

__global__ void vecMul(float *a, float *b, float *c, int n)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        c[tid] = a[tid] * b[tid];
    }
}


__global__ void copy(float *dst, float *src, int n)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        dst[tid] = src[tid];
    }
}

__global__ void update(float *v, float *dC_dv, float eta, int n)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < n){
        v[tid] -= eta * dC_dv[tid];
    }
}

__global__ void cal_dw(float *a, float *delta, float *dC_dw, int input_size, int output_size)
{
    // dC_dw(j, k) = a[l-1](k) * delta[j]
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < input_size * output_size){
        dC_dw[tid] = delta[tid / input_size] * a[tid % input_size];
    }
}

__global__ void cal_loss(float *a, float *y, float *loss, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int sidx = threadIdx.x;

    float diff = (a[tid] - y[tid]);
    sdata[sidx] = 0.5 * diff * diff;

    __syncthreads();

    // Reduce within the block using shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (sidx < s) {
            sdata[sidx] += sdata[sidx + s];
        }
        __syncthreads();
    }

    // Atomic add to accumulate the block sums
    if (sidx == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

