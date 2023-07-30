//
// Created by JunchengJi on 7/30/2023.
//

#include "MLP.h"

MLP::MLP(int num_layers, std::vector<int> layers) {
    this->num_layers = num_layers;
    for (int i = 0; i < num_layers - 1; ++i) {
        this->layers.push_back(new Dense(layers[i], layers[i + 1], Activation::ActivationFunctionType::SIGMOID));
    }
    total_loss = 0;
    output_dim = layers[num_layers - 1];
}

double* MLP::forward(double *input_data) {
    double *output = input_data;
    for (int i = 0; i < num_layers - 1; ++i) {
        output = layers[i]->forward(output);
    }
    return output;
}

double *MLP::backward(double *grad) {
    double *output = grad;
    for (int i = num_layers - 2; i >= 0; --i) {
        output = layers[i]->backward(output);
    }
    return output;
}

void MLP::update(double lr, int batchSize) {
    for (int i = 0; i < num_layers - 1; ++i) {
        layers[i]->update(lr, batchSize);
    }
}

MLP::~MLP() {
    for (int i = 0; i < num_layers - 1; ++i) {
        delete layers[i];
    }
}

void MLP::loss(const double *output, const double *label) {
    for(int i = 0; i < output_dim; ++i) {
        total_loss += (output[i] - label[i]) * (output[i] - label[i]) * 0.5;
    }
}

void MLP::shuffle(std::vector<double*> &x, std::vector<double*> &y)
{
    std::random_device rd;
    std::mt19937 g(rd());
    for(int i = 0; i < x.size(); ++i) {
        int k = std::uniform_int_distribution<int>(0, x.size() - 1)(g);
        std::swap(x[i], x[k]);
        std::swap(y[i], y[k]);
    }
}

void MLP::train(int epoch, int batch_size, double lr, std::vector<double *> &train_data,
                std::vector<double *> &train_label) {
    for (int i = 0; i < epoch; ++i) {
        std::cout << "epoch: " << i << " ....";

        shuffle(train_data, train_label);
        total_loss = 0;
        for (int j = 0; j < train_data.size(); j += batch_size) {
            for(int k = 0; k < batch_size; ++k) {
                double *output = forward(train_data[j + k]);
                loss(output, train_label[j + k]);
                auto grad = new double[output_dim];
                for (int l = 0; l < output_dim; ++l) {
                    grad[l] = output[l] - train_label[j + k][l];
                }
                backward(grad);
                delete[] grad;
            }
            update(lr, batch_size);
        }
        std::cout << " loss: " << total_loss/(double)train_data.size() << std::endl;
    }
}

void MLP::predict(std::vector<double *> &test_data, std::vector<int> &test_label) {
    int correct = 0;
    int max_idx = 0;
    for(int i = 0; i < test_data.size(); ++i) {
        double *output = forward(test_data[i]);
        double max = output[0];
        for(int j = 1; j < output_dim; ++j) {
            if(output[j] > max) {
                max = output[j];
                max_idx = j;
            }
        }
        if(max_idx == test_label[i]) {
            correct++;
        }
    }
    std::cout << "accuracy: " << (double)correct/(double)test_data.size() << std::endl;
}