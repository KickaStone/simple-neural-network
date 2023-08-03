//
// Created by JunchengJi on 7/31/2023.
//

#ifndef NETWORK_CNN_H
#define NETWORK_CNN_H

#include "layers/layer.h"
#include "net.h"
#include <Eigen/Dense>

class CNN : public Net{
private:
    std::vector<Layer*> _layers;
    std::vector<double*> _train_data;
    std::vector<int> _train_label;
    std::vector<double*> _test_data;
    std::vector<int> _test_label;

public:
    ~CNN();
    CNN(int num_layers, int output_dim);
    void addLayer(Layer* layer);

    void setDataset(std::vector<double*> &train_data, std::vector<int> &train_label,
                    std::vector<double*> &test_data, std::vector<int> &test_label);
    double* forward(double *input_data) override;
    double* backward(double *grad) override;
    void update(double lr, int batchSize) override;
    bool isValid();
    void train(int epoch, double lr, int batchSize);
    double predict();
};


#endif //NETWORK_CNN_H
