//
// Created by JunchengJi on 7/30/2023.
//
#include "dense.h"

Dense::Dense(int input_size, int output_size, Activation::ActivationFunctionType type) : Layer(input_size, output_size, type) {
    this->input = nullptr;
    this->w = new double[input_size * output_size];
    this->b = new double[output_size];
    this->a = new double[output_size];
    this->dz = new double[output_size];
    this->dw = new double[input_size * output_size];
    this->db = new double[output_size];
    this->input_grad = new double[input_size];

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
    delete[] this->input_grad;
}

double* Dense::forward(const double *input_data) {
    this->input = input_data;
    for(int i = 0; i < outputSize; i++) {
        a[i] = b[i];
        for (int j = 0; j < inputSize; j++) {
            a[i] += w[i * inputSize + j] * input_data[j];
        }
        a[i] = activation(a[i]);
    }
    return a;
}

double* Dense::backward(const double *output_grad) {
    for(int i = 0; i < outputSize; i++){
        dz[i] = output_grad[i] * derivative(a[i]);
    }

    // dw
    for(int i = 0; i < outputSize; i++){
        for(int j = 0; j < inputSize; j++){
            dw[i * inputSize + j] += dz[i] * input[j];
        }
    }

    // db
    for (int i = 0; i < outputSize; ++i) {
        db[i] += dz[i];
    }

    // input layer's dC/da
    for(int j = 0; j < inputSize; j++){
        input_grad[j] = 0;
        for(int i = 0; i < outputSize; i++){
            input_grad[j] += dz[i] * w[i * inputSize + j];
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

void Dense::save() {
    // get current time
    time_t now = time(nullptr);
    tm *ltm = localtime(&now);
    std::string time = std::to_string(1900 + ltm->tm_year) + std::to_string(1 + ltm->tm_mon) + std::to_string(ltm->tm_mday) + std::to_string(ltm->tm_hour) + std::to_string(ltm->tm_min) + std::to_string(ltm->tm_sec);

    std::string w_file = "w_" + std::to_string(outputSize) + "_" + time + ".txt";
    std::string b_file = "b_" + std::to_string(outputSize) + "_" + time + ".txt";

    FILE *fp_w = fopen(w_file.c_str(), "w");
    FILE *fp_b = fopen(b_file.c_str(), "w");

    for(int i = 0; i < outputSize; i++){
        for(int j = 0; j < inputSize; j++){
            fprintf(fp_w, "%f ", w[i * inputSize + j]);
        }
        fprintf(fp_w, "\n");
        fprintf(fp_b, "%f\n", b[i]);
    }

    fclose(fp_w);
    fclose(fp_b);
}

