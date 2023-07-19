#include "Network.h"
#include "Mnist_helper.h"
#include <iostream>

// #define TRAIN_NETWORK
#define TEST_NETWORK

void test(){

    Network *network;

    mnist_data *training_data;
    mnist_data *test_data;

    unsigned int num_training_data = 0;
    unsigned int num_test_data = 0;

    char* training_data_file = "E:/projects/Cuda/Network/data/train-images-idx3-ubyte";
    char* training_label_file = "E:/projects/Cuda/Network/data/train-labels-idx1-ubyte";
    char* test_data_file = "E:/projects/Cuda/Network/data/t10k-images-idx3-ubyte";
    char* test_label_file = "E:/projects/Cuda/Network/data/t10k-labels-idx1-ubyte";

    int ret = mnist_load(training_data_file, training_label_file, &training_data, &num_training_data);
    if(ret != 0){
        std::cout << "Error loading training data" << std::endl;
        return;
    }

    ret = mnist_load(test_data_file, test_label_file, &test_data, &num_test_data);
    if(ret != 0){
        std::cout << "Error loading test data" << std::endl;
        return;
    }

    // convert
    double **data = (double**)malloc(sizeof(double*) * num_training_data);
    unsigned int *label = (unsigned int *)malloc(sizeof(unsigned int) * num_training_data);

    for(int i = 0; i < num_training_data; i++){
        data[i] = (double*)malloc(sizeof(double) * 784);
    }
    
    double **t_data = (double**)malloc(sizeof(double*) * num_test_data);
    unsigned int *t_label = (unsigned int *)malloc(sizeof(unsigned int) * num_test_data);

    for(int i = 0; i < num_test_data; i++){
        t_data[i] = (double*)malloc(sizeof(double) * 784);
    }

    convert(data, label, training_data, num_training_data);
    convert(t_data, t_label, test_data, num_test_data);

#ifdef TRAIN_NETWORK    
    // create network
    int sizes[] = {784, 30, 10};
    int num_layers = 3;
    
    network = new Network(sizes, num_layers);

    // train network
    network->SGD(data, label, 30, 10, 3.0, t_data, t_label, num_training_data, num_test_data);

    // save network
    network->saveNetwork();

    // free memory
    for(int i = 0; i < num_training_data; i++){
        free(data[i]);
    }

    free(data);
    free(label);
    
    for(int i = 0; i < num_test_data; i++){
        free(t_data[i]);
    }

    free(t_data);
    free(t_label);

#else

    #ifdef TEST_NETWORK
    
    // load network
    network = new Network();
    network->loadNetwork();

    // test network


    #else
    // do nothing
    printf(
        "Nothing to do.\n"
    );

    #endif

#endif
}

