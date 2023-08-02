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

void MLP::shuffle(std::vector<double*> &x, std::vector<int> &y)
{
    std::random_device rd;
    std::mt19937 g(rd());
    for(int i = 0; i < (int)x.size(); ++i) {
        int k = std::uniform_int_distribution<int>(0, x.size() - 1)(g);
        std::swap(x[i], x[k]);
        std::swap(y[i], y[k]);
    }
}

void MLP::train(int epoch, int batch_size, double lr){
    for (int e = 0; e < epoch; ++e) {
        std::cout << "epoch: " << e << " ....";
        shuffle(x, y);
        total_loss = 0;
        for (int j = 0; j < (int)x.size(); j += batch_size) {
            // std:: cout << "batch: " << j/batch_size << std::endl;
            for(int k = 0; k < batch_size; ++k) {
                double *output = forward(x[j + k]);
                auto grad = new double[output_dim];
                for(int i = 0; i < output_dim; ++i) {
                    int y_i = y[j + k] == i ? 1 : 0;
                    grad[i] = output[i] - y_i;
                    total_loss += grad[i] * grad[i] * 0.5;
                }
                backward(grad);
                delete[] grad;
            }
            update(lr, batch_size);
        }
        std::cout << " loss: " << total_loss << " ";
        predict(t_x, t_y);
    }
}

void MLP::predict(std::vector<double *> &test_data, std::vector<int> &test_label) {
    int correct = 0;
    int max_idx = 0;
    double loss = 0.0;
    double max = 0.0;
    for(int i = 0; i < (int)test_data.size(); ++i) {
        double *output = forward(test_data[i]);
        max_idx = 0;
        max = output[0];
        for(int j = 0; j < output_dim; ++j) {
            double y = test_label[i] == j ? 1 : 0;
            loss += (output[j] - y) * (output[j] - y) * 0.5;
            if(output[j] > max) {
                max = output[j];
                max_idx = j;
            }
        }
        if(max_idx == test_label[i])
            correct++;
    }
    std::cout << "accuracy: " << (double)correct/(double)test_data.size() * 100 << "% test_loss: " << loss << std::endl;
}

void MLP::set_dataset(std::vector<double *> &x, std::vector<int> &y, std::vector<double *> &t_x, std::vector<int> &t_y)
{
    this->x = x;
    this->y = y;
    this->t_x = t_x;
    this->t_y = t_y;
}
