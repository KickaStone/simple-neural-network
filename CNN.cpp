//
// Created by JunchengJi on 7/31/2023.
//
#include "CNN.h"

CNN::CNN(int num_layers, int output_dim) {
    this->num_layers = num_layers;
    this->output_dim = output_dim;
}

CNN::~CNN() {
    for(auto layer : layers){
        delete layer;
    }
}

void CNN::addLayer(Layer *layer) {
    layers.push_back(layer);
}

double* CNN::forward(double *input_data) {
    double *input = input_data;
    for(auto &layer : layers){
        input = layer->forward(input);
    }
    return input;
}

double* CNN::backward(double *output_grad) {
    double *input = output_grad;
    for(int i = (int)layers.size() - 1; i >= 0; i--){
        input = layers[i]->backward(input);
    }
    return input;
}

void CNN::update(double lr, int batchSize) {
    for(auto layer : layers){
        layer->update(lr, batchSize);
    }
}

void CNN::train(std::vector<double *> &input_data, std::vector<double *> &label, std::vector<double*> &test_data, std::vector<int> &test_label, int epoch, double lr, int batchSize) {
    // SGD
    for(int e = 1; e <= epoch; e++){
        std::cout << "Epoch: " << e << std::endl;
        // shuffle data
        std::random_device rd;
        std::mt19937 g(rd());
        for(int i = 0; i < (int)input_data.size(); i++){
            int j = std::uniform_int_distribution<int>(i, input_data.size() - 1)(g);
            std::swap(input_data[i], input_data[j]);
            std::swap(label[i], label[j]);
        }
        double loss = 0;
        // mini-batch
        for(int i = 0; i < (int)input_data.size(); i += batchSize){
//            std::cout << "batch: " << i / batchSize << std::endl;
            for(int j = 0; j < batchSize; j++){
                auto input = forward(input_data[i + j]);
                auto *grad = new double[output_dim];
                for(int k = 0; k < output_dim; k++){
                    grad[k] = input[k] - label[i+j][k];
                    loss += 0.5 * grad[k] * grad[k];
                }
//                std::cout << "loss: " << loss << std::endl;
                backward(grad);
                delete[] grad;
            }
            update(lr, batchSize);
        }
        loss /= (double)input_data.size();
        std::cout << "Epoch: " << e << " Loss: " << loss << std::endl;
        predict(test_data, test_label);
    }
}

void CNN::predict(std::vector<double *> &data, std::vector<int> &labels) {
    int correct = 0;
    int max_idx = 0;
    for(int i = 0; i < (int)data.size(); i++){
        double *input = forward(data[i]);
        max_idx = 0;
        double max = input[0];
        for(int j = 0; j < output_dim; j++){
            if(input[j] > max){
                max = input[j];
                max_idx = j;
            }
        }
        if(max_idx == labels[i]){
            correct++;
        }
    }
    std::cout << "Accuracy: " << (double)correct / (double)data.size() << std::endl;
}
