//
// Created by JunchengJi on 7/31/2023.
//

#ifndef NETWORK_CNN_H
#define NETWORK_CNN_H

#include "layers/layer.h"
#include "net.h"

class CNN : public Net{
private:
    std::vector<Layer*> layers;
public:
    CNN() = default;
    ~CNN();
    CNN(int num_layers, int output_dim);
    void addLayer(Layer* layer);

    double* forward(double *input_data) override;
    double* backward(double *grad) override;
    void update(double lr, int batchSize) override;

    void train(std::vector<double*> &input_data, std::vector<double*> &label, std::vector<double*> &test_data, std::vector<int> &test_label, int epoch, double lr, int batchSize);
    void predict(std::vector<double*> &data, std::vector<int> &labels);
};


#endif //NETWORK_CNN_H
