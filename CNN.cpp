//
// Created by JunchengJi on 7/31/2023.
//
#include "CNN.h"

CNN::CNN(int num_layers, int output_dim){
    this->num_layers = num_layers;
    this->output_dim = output_dim;
}

CNN::~CNN() {
    for(auto layer : _layers){
        delete layer;
    }
}

void CNN::addLayer(Layer *layer) {
    _layers.push_back(layer);
}

void CNN::setDataset(std::vector<double *> &train_data, std::vector<int> &train_label, std::vector<double *> &test_data, std::vector<int> &test_label)
{
    _train_data = train_data;
    _train_label = train_label;
    _test_data = test_data;
    _test_label = test_label;
}

double *CNN::forward(double *input_data)
{
    double *input = input_data;
    for(auto &layer : _layers){
        input = layer->forward(input);
    }
    return input;
}

double* CNN::backward(double *grad) {
    double *input = grad;
    for(int i = (int)_layers.size() - 1; i >= 0; i--){
        input = _layers[i]->backward(input);
    }
    return input;
}

void CNN::update(double lr, int batchSize) {
    for(auto layer : _layers){
        layer->update(lr, batchSize);
    }
}

bool CNN::checkVaildity()
{
    if(_layers.empty()){
        return false;
    }
    for(int i = 0; i < (int)_layers.size() - 1; i++){
        std::cout << _layers[i]->getOutputSize() << " " << _layers[i+1]->getInputSize() << std::endl;
        if(_layers[i]->getOutputSize() != _layers[i+1]->getInputSize()){
            return false;
        }
    }
    return true;
}

void shuffle(std::vector<double *> &input_data, std::vector<int> &label) {
    std::random_device rd;
    std::mt19937 g(rd());
    for(int i = 0; i < (int)input_data.size(); i++){
        int j = std::uniform_int_distribution<int>(i, input_data.size() - 1)(g);
        std::swap(input_data[i], input_data[j]);
        std::swap(label[i], label[j]);
    }
}

void CNN::train(int epoch, double lr, int batchSize) {
    if(!checkVaildity()){
        throw std::runtime_error("Invalid network");
    }
    // SGD
    for(int e = 1; e <= epoch; e++){
        std::cout << "Epoch: " << e << std::endl;
        // shuffle data
        shuffle(_train_data, _train_label);
        double loss = 0;
        // mini-batch SGD
        for(int i = 0; i < (int)_train_data.size(); i += batchSize){
            for(int j = 0; j < batchSize; j++){
                auto input = forward(_train_data[i + j]);
                auto *grad = new double[output_dim];
                for(int k = 0; k < output_dim; k++){
                    double y = _train_label[i+j] == k ? 1.0 : 0.0;
                    grad[k] = input[k] - y;
                    loss += 0.5 * grad[k] * grad[k];
                }
                backward(grad);
                delete[] grad;
            }
            update(lr, batchSize);
        }
        loss /= (double)_train_data.size();
        std::cout << "Epoch: " << e << " Loss: " << loss << std::endl;
        predict();
    }
}

void CNN::predict() {
    if(_test_data.empty() || _test_label.empty()){
        return;
    }
    int correct = 0;
    int max_idx = 0;
    double test_loss = 0.0;
    for(int i = 0; i < (int)_test_data.size(); i++){
        double *input = forward(_test_data[i]);
        max_idx = 0;
        double max = input[0];
        for(int j = 0; j < output_dim; j++){
            if(input[j] > max){
                max = input[j];
                max_idx = j;
            }
            if(j == _test_label[i]){
                test_loss += 0.5 * (input[j] - 1.0) * (input[j] - 1.0);
            }else{
                test_loss += 0.5 * input[j] * input[j];
            }
        }
        if(max_idx == _test_label[i]){
            correct++;
        }
    }
    std::cout << "Accuracy: " << (double)correct * 100 / (double)_test_data.size() << "%, test_loss: " << test_loss << std::endl;
}
