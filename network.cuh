#pragma once

#include <vector>
#include <numeric>
#include <iostream>

#include <curand.h>
#include <random>
#include <cuda_runtime.h>

#include "spdlog/spdlog.h"
#include "mathkernel.cuh"
#include "common.h"

#define DEBUG

#ifdef USE_FLOAT
#define DATA_TYPE float
#else
#define DATA_TYPE double
#endif

struct Params
{
    std::vector<int> layers;
    int inputsize;
    int outputsize;
    /**
     * @brief number of layers, not including input layer
    */
    int num_layers;
    int batchsize;
    DATA_TYPE eta;
    static const int blocksize = 256;
};


class NeuralNetwork
{
    Params params;
    int blocks;

    std::vector<float*> input;
    std::vector<int> labal;
    float *data;
    float *y;
    float loss;

    std::vector<float*> w; // weights
    std::vector<float*> b; // biases
    std::vector<float*> z; // z = w * a + b
    std::vector<float*> a; // a = activation(z)

    std::vector<float*> z_prime; // activation'(z)
    std::vector<float*> dC_dz; // delta = dc_dz = dc_da * da_dz = dc_da * activation'(z)
    std::vector<float*> dC_da; // dc_da = dc_dz * dz_da = dc_dz * w
    std::vector<float*> dC_dw; // dc_dw = dc_dz * dz_dw = dc_dz * a
    std::vector<float*> dC_db; // dc_db = dc_dz * dz_db = dc_dz * 1

    std::vector<float*> batch_dw;
    std::vector<float*> batch_db;

    void fillRandom(float* arr, int size);
    void fillZero(float* arr, int size);

public:

    /// @brief initialize the neural network
    /// @param input input data dim
    /// @param shape shape of the neural network, does not include input layer
    /// @param eta learning rate
    NeuralNetwork(int input, std::vector<int> shape, float eta);
    ~NeuralNetwork();

    
    // void train(float* input, int labal);

    /**
     * @brief forward propagation
     * @param input input data
     * @param size size of input data
    */
    float* forward(float* input, int size);

    /**
     * @brief backpropagation
     * @param y the expected output
    */
    void setParams(float learning_rate, int batch_size);
    void backprop(float* y);
    void update_weights_and_biases();
    float getLoss(float* y);
    void SDG_train(std::vector<float*> &training_data, std::vector<float*> training_label, int epochs, int batch_size, std::vector<float*> &test_data, std::vector<int> &test_label);
    void mini_batch(std::vector<float*> &training_data, std::vector<float*> &training_label, int batch_size, int start);
    // int predict(float* input);
    // void etaUpdate(float eta);

    #ifdef DEBUG
    Params _debug_params();
    void _debug_get_a(std::vector<float*> a);
    void _debug_get_delta(std::vector<float*> delta);
    void _debug_set(std::vector<float*> w, std::vector<float*> b);
    void _debug_get_weights_and_biases(std::vector<float*> w, std::vector<float*> b);
    void _debug_get_grad(std::vector<float*> dC_dw, std::vector<float*> dC_db);
    #endif
};