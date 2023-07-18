#ifndef __NETWORK_H__
#define __NETWORK_H__


#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#define MNIST_STATIC

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <time.h>
#include <random>

struct nabla{
    double **nabla_bias;
    double **nabla_weight;

    nabla(){
        nabla_bias = NULL;
        nabla_weight = NULL;
    }

    nabla(const nabla& rhs){
        nabla_bias = rhs.nabla_bias;
        nabla_weight = rhs.nabla_weight;
    }
};

class Network{
public:
    int num_layers;
    int* sizes;
    
    Network();
    
    Network(int* sizes, int num_layers);

    double getBias(int layer, int index);
    double getWeight(int layer, int j, int k);

    void setBias(int layer, int index, double value);
    void setWeight(int layer, int j, int k, double value);

    double* feedforward(const double* input);

    double cost_derivative(double output, double y);

    nabla backprop(const double* input, unsigned int label);

    int evaluate(double **data, unsigned int *label, int num_data); /* new */

    void update_mini_batch(double **data, unsigned int *label, int start, int mini_batch_size, double eta); /* new */

    void SGD(double **data, unsigned int *label, int epochs, int mini_batch_size, double eta, double **test_data, unsigned int *test_label, int num_training_data, int num_test_data); /* new */
    
    ~Network();
    void print();

    void saveNetwork();
    void loadNetwork();
    
private:
    double **bias;
    double **weight;
    bool checkBiasIdx(int layer, int index);
    bool checkWeightIdx(int layer, int j, int k);
    
    void init();

    
};



#endif // __NETWORK_H__
