#include <gtest/gtest.h>
#include "NetworkTest.h"

#define LOAD_NETWORK

#ifdef LOAD_NETWORK
    #undef LOAD_NETWORK
class NetworkTest:public::testing::Test{
protected:
    void SetUp() override{
        int sizes[] = {784, 30, 10};
        network = new Network(sizes, 3);
        
    }

    Network* network;
    mnist_data *training_data;
    mnist_data *test_data;
};

TEST_F(NetworkTest, testLoad){
    char* training_data_file = "E:/projects/Cuda/Network/data/train-images-idx3-ubyte";
    char* training_label_file = "E:/projects/Cuda/Network/data/train-labels-idx1-ubyte";
    char* test_data_file = "E:/projects/Cuda/Network/data/t10k-images-idx3-ubyte";
    char* test_label_file = "E:/projects/Cuda/Network/data/t10k-labels-idx1-ubyte";

    unsigned int num_training_data = 0;
    unsigned int num_test_data = 0;
    int ret = mnist_load(training_data_file, training_label_file, &training_data, &num_training_data);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(num_training_data, 60000);

    ret = mnist_load(test_data_file, test_label_file, &test_data, &num_test_data);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(num_test_data, 10000);

    // convert
    double **data = new double*[num_training_data];
    unsigned int *label = new unsigned int[num_training_data];
    for(int i = 0; i < num_training_data; i++){
        data[i] = new double[784];
    }

    double **t_data = new double*[num_test_data];
    unsigned int *t_label = new unsigned int[num_test_data];
    for(int i = 0; i < num_test_data; i++){
        t_data[i] = new double[784];
    }
    convert(data, label, training_data, num_training_data);
    convert(t_data, t_label, test_data, num_test_data);

    double *output = network->feedforward(t_data[0]);
    // 
    for(int i = 0; i < 30; i++){
        printf("%f \n", network->getWeight(2, 0, i));
    }


    network->SGD(data, label, 30, 10, 3.0, t_data, t_label, num_training_data, num_test_data);

    network->saveNetwork();

    output = network->feedforward(t_data[0]);
    for(int i = 0; i < 30; i++){
        printf("%f \n", network->getWeight(2, 0, i));
    }
}

#else

class NetworkTest:public::testing::Test{
protected:

    void SetUp() override{
        network = new Network();
        network->loadNetwork();
    }

    mnist_data *test_data;
    double **data;
    unsigned int *label;
    Network *network;
};

TEST_F(NetworkTest, testLoad){
    char* test_data_file = "E:/projects/Cuda/Network/data/t10k-images-idx3-ubyte";
    char* test_label_file = "E:/projects/Cuda/Network/data/t10k-labels-idx1-ubyte";

    unsigned int num_test_data = 0;
    int ret = mnist_load(test_data_file, test_label_file, &test_data, &num_test_data);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(num_test_data, 10000);

    // convert
    data = new double*[num_test_data];
    label = new unsigned int[num_test_data];
    for(int i = 0; i < num_test_data; i++){
        data[i] = new double[784];
    }
    convert(data, label, test_data, num_test_data);
    for(int k = 0; k < 100; k++){
        // print image
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                if(data[k][i * 28 + j] > 0.1){
                    printf("0");
                }else{
                    printf(" ");
                }
            }
            printf("\n");
        }

        double *output = network->feedforward(data[k]);
        
        // get maxoutput indx
        int max_idx = 0;
        for(int i = 0; i < 10; i++){
            if(output[i] > output[max_idx]){
                max_idx = i;
            }
        }
        printf("output: %d || label: %d\n", max_idx, label[k]);
        if(max_idx == label[k]){
            printf("@@@@@@@@@@@@@@@@@\n");
        }
        else{
            printf("!!!!!!!!!!!!!!!!\n");
        }
    }
}
#endif