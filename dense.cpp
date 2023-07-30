//
// Created by JunchengJi on 7/30/2023.
//
#include "dense.h"

Dense::Dense(int input_size, int output_size, Activation::ActivationFunctions af) : Layer(input_size, output_size, af) {
    this->input = nullptr;
    this->w = new double[input_size * output_size];
    this->b = new double[output_size];
    this->a = new double[output_size];
    this->dz = new double[output_size];
    this->dw = new double[input_size * output_size];
    this->db = new double[output_size];

    // initialize w and b
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    for(int i = 0; i < input_size * output_size; i++){
        w[i] = d(gen);
    }
    std::fill(b, b + output_size, 0);
    std::fill(dw, dw + input_size * output_size, 0);
    std::fill(db, db + output_size, 0);
}

Dense::~Dense() {
    delete[] this->w;
    delete[] this->b;
    delete[] this->a;
    delete[] this->db;
    delete[] this->dw;
    delete[] this->dz;
}

double* Dense::forward(double *input_data) {
    this->input = input_data;
    for(int i = 0; i < outputSize; i++) {
        a[i] = b[i];
        for (int j = 0; j < inputSize; j++) {
            a[i] += w[i * inputSize + j] * input_data[j];
        }
        a[i] = activationFunc.activation(a[i]);
    }
    return a;
}

double* Dense::backward(double *output_grad) {
    // output layer's dC/dz
    for(int i = 0; i < outputSize; i++){
        dz[i] = output_grad[i] * activationFunc.derivative(a[i]);
    }

    for(int i = 0; i < inputSize; i++){
        for(int j = 0; j < outputSize; j++){
            dw[j * inputSize + i] += dz[j] * input[i];
        }
    }

    for (int i = 0; i < outputSize; ++i) {
        db[i] += dz[i];
    }

    // input layer's dC/da
    auto *input_grad = new double[inputSize];
    for(int i = 0; i < inputSize; i++) {
        input_grad[i] = 0;
        for (int j = 0; j < outputSize; j++) {
            input_grad[i] += dz[j] * w[j * inputSize + i];
        }
    }
    return input_grad;
}

void Dense::update(double lr, int batchSize) {
    for(int i = 0; i < inputSize; i++){
        for(int j = 0; j < outputSize; j++){
            w[j * inputSize + i] -= (lr / batchSize) * dw[j * inputSize + i];
        }
    }

    for(int i = 0; i < outputSize; i++){
        b[i] -= (lr / batchSize) * db[i];
    }

    std::fill(dw, dw + inputSize * outputSize, 0);
    std::fill(db, db + outputSize, 0);
}

