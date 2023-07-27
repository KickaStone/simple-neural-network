#pragma once

#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>

#include <curand.h>
#include <random>
#include <cuda_runtime.h>

#include "common.h"

#define DEBUG
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
    int epochs;
    double eta;
    static const int blocksize = 256;
};


class NeuralNetwork
{
    Params params;
    int blocks;
    cublasHandle_t cublasH;

    std::vector<double*> input;
    std::vector<int> labal;
    double *data;
    double *y;
    double loss;

    std::vector<double*> w; // weights
    std::vector<double*> b; // biases
    std::vector<double*> z; // z = w * a + b
    std::vector<double*> a; // a = activation(z)

    std::vector<double*> z_prime; // activation'(z)
    std::vector<double*> dC_dz; // delta = dc_dz = dc_da * da_dz = dc_da * activation'(z)
    std::vector<double*> dC_da; // dc_da = dc_dz * dz_da = dc_dz * w
    std::vector<double*> dC_dw; // dc_dw = dc_dz * dz_dw = dc_dz * a
    std::vector<double*> dC_db; // dc_db = dc_dz * dz_db = dc_dz * 1

    void fillRandom(double* arr, int size);
    void fillZero(double* arr, int size);

public:

    /// @brief initialize the neural network
    /// @param input input data dim
    /// @param shape shape of the neural network, does not include input layer
    /// @param eta learning rate
    NeuralNetwork(int input, std::vector<int> shape);
    ~NeuralNetwork();

    
    // void train(double* input, int labal);

    /**
     * @brief forward propagation
     * @param input input data
     * @param size size of input data
    */
    double* forward(double* input, int size);

    /**
     * @brief backpropagation
     * @param y the expected output
    */
    void setParams(double learning_rate, int batch_size, int epochs);
    void backprop(double* y);
    void update_weights_and_biases();
    void train(std::vector<double*> &training_data, std::vector<double*> training_label, std::vector<double*> &test_data, std::vector<int> &test_label);
    void mini_batch(std::vector<double*> &training_data, std::vector<double*> &training_label, int batch_size, int start);
    void evaluate(std::vector<double*> &test_data, std::vector<int> &test_label, int &correct, double &loss);
    void save(); // save the weights and biases

    #ifdef DEBUG
    Params _debug_params();
    void _debug_get_a(std::vector<double*> a);
    void _debug_get_delta(std::vector<double*> delta);
    void _debug_set(std::vector<double*> w, std::vector<double*> b);
    void _debug_get_weights_and_biases(std::vector<double*> w, std::vector<double*> b);
    void _debug_get_grad(std::vector<double*> dC_dw, std::vector<double*> dC_db);
    #endif
};