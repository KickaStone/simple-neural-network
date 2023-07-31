//
// Created by JunchengJi on 7/31/2023.
//
#include "CNN.h"

CNN::CNN(int num_layers, int output_dim) {
    this->num_layers = num_layers;
    this->output_dim = output_dim;
}

CNN::~CNN() {
    for(auto &layer : layers){
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

double* CNN::backward(double *grad) {
    double *input = grad;
    for(auto &layer : layers){
        input = layer->backward(input);
    }
    return input;
}

void CNN::update(double lr, int batchSize) {
    for(auto &layer : layers){
        layer->update(lr, batchSize);
    }
}

void CNN::train(std::vector<double *> &input_data, std::vector<double *> &label, int epoch, double lr, int batchSize) {
    // SGD
    for(int e = 1; e <= epoch; e++){
        std::cout << "Epoch: " << e << std::endl;
        // shuffle data
        std::random_device rd;
        std::mt19937 g(rd());
        for(int i = 0; i < input_data.size(); i++){
            int j = std::uniform_int_distribution<int>(i, input_data.size() - 1)(g);
            std::swap(input_data[i], input_data[j]);
            std::swap(label[i], label[j]);
        }
        double loss = 0;
        std::cout << "Batch Size: " << batchSize << std::endl;
        // mini-batch
        for(int i = 0; i < input_data.size(); i += batchSize){
            double *input = input_data[i];
            double *label_data = label[i];
            for(int j = 0; j < batchSize; j++){
                input = forward(input_data[i + j]);
            }
            auto *grad = new double[output_dim];
            for(int j = 0; j < output_dim; j++){
                grad[j] = input[j] - label_data[j];
                loss += grad[j] * grad[j];
            }
            backward(grad);
            update(lr, batchSize);
        }
        std::cout << "Epoch: " << e << " Loss: " << loss << std::endl;
    }
}

void CNN::predict(std::vector<double *> &data, std::vector<int> &labels) {
    int correct = 0;
    int max_idx = 0;
    for(int i = 0; i < data.size(); i++){
        double *input = forward(data[i]);
        double max = input[0];
        for(int j = 1; j < output_dim; j++){
            if(input[j] > max){
                max = input[j];
                max_idx = j;
            }
        }
        if(max_idx == labels[i]){
            correct++;
        }
    }
    std::cout << "Accuracy: " << correct / data.size() << std::endl;
}
