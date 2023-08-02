//
// Created by JunchengJi on 7/30/2023.
//

#ifndef NETWORK_MLP_H
#define NETWORK_MLP_H

#include <iostream>
#include <vector>
#include "layers/dense.h"
#include "Net.h"

class MLP : Net {
public:
    MLP(int num_layers, std::vector<int> layers);
    ~MLP();
    double* forward(double *input_data) override;
    double* backward(double *grad) override;
    void update(double lr, int batchSize) override;
    void train(int epoch, int batch_size, double lr);
    void predict(std::vector<double*>& test_data, std::vector<int>& test_label);
    void set_dataset(std::vector<double*>& x, std::vector<int>& y, std::vector<double*>& t_x, std::vector<int>& t_y);

private:
    double total_loss = 0;
    std::vector<double*> x, t_x;
    std::vector<int> y, t_y;
    void loss(const double *output, const double *label);
    void shuffle(std::vector<double*>& x, std::vector<int>& y);    
};


#endif //NETWORK_MLP_H
