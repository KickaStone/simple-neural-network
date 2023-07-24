#pragma once

#define CUDA_CHECK(call) \
{ \
    cudaError_t err = call; \
    if (cudaSuccess != err) \
    { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}