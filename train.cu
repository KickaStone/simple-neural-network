#include <stdio.h>
#include <cuda_runtime.h>

#define USE_MNIST_LOADER
#include "mnist.h"
#include <random>
#include <math_functions.h>
#include <cuda_runtime.h>

#include "kernel.cuh"

// #define DEBUG

void checkCudaError(cudaError_t err, const char* file, const int line){
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK(err) (checkCudaError(err, __FILE__, __LINE__))

int main(int argc, char const *argv[])
{
    cudaSetDevice(0);    

    // load mnist data
    mnist_data *data1, *data2;
    unsigned int cnt1, cnt2;
    int ret1, ret2;
    const char image_file[] = "../data/train-images-idx3-ubyte";    
    const char image_label[] = "../data/train-labels-idx1-ubyte";
    const char test_image_file[] = "../data/t10k-images-idx3-ubyte";
    const char test_image_label[] = "../data/t10k-labels-idx1-ubyte";

    ret1 = mnist_load(image_file, image_label, &data1, &cnt1);
    ret2 = mnist_load(test_image_file, test_image_label, &data2, &cnt2);
    if (ret1 != 0 || ret2 != 0) {
        printf("Error loading mnist data\n");
        return -1;
    }else{
        printf("Success loading mnist data, data size: %d | %d\n", cnt1, cnt2);
    }

    float **training_data = (float **)malloc(sizeof(float *) * cnt1);
    float **target = (float **)malloc(sizeof(float *) * cnt1);
    float **test_data = (float **)malloc(sizeof(float *) * cnt2);
    int *test_label = (int *)malloc(sizeof(int) * cnt2);
    for (int i = 0; i < cnt1; i++)
    {
        training_data[i] = (float *)malloc(sizeof(float) * 784);
        target[i] = (float *)malloc(sizeof(float) * 10);
    }
    

    for(int i = 0; i < cnt1; i++){
        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                training_data[i][x*28+y] = (float)data1[i].data[x][y] / 255.0f;

        for (int j = 0; j < 10; j++)
            target[i][j] = 0.0f;

        target[i][data1[i].label] = 1.0f;       
    }

    for(int i = 0; i < cnt2; i++){
        test_data[i] = (float *)malloc(sizeof(float) * 784);
        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++)
                test_data[i][x*28+y] = (float)data2[i].data[x][y] / 255.0f;

        test_label[i] = data2[i].label;
    }

    // create network

    int n = 3;
    int *layers = (int *)malloc(sizeof(int) * n);
    layers[0] = 784;
    layers[1] = 30;
    layers[2] = 10;

    float **h_weights;
    float **h_biases;
    float **d_weights;
    float **d_biases;


    h_weights = (float **)malloc(sizeof(float *) * n);  
    h_biases = (float **)malloc(sizeof(float *) * n);

    for(int i = 1; i < n; i++){
        h_weights[i] = (float *)malloc(sizeof(float) * layers[i] * layers[i-1]);
        h_biases[i] = (float *)malloc(sizeof(float) * layers[i]);
    }

    // initialize weights and biases
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 1.0f);
    for(int i = 1; i < n; i++){
        for(int j = 0; j < layers[i]; j++){
            for(int k = 0; k < layers[i-1]; k++){
                #ifdef DEBUG
                    h_weights[i][j*layers[i-1]+k] = 0.0f;
                #else
                    h_weights[i][j*layers[i-1]+k] = d(gen);
                #endif
            }
            #ifdef DEBUG
            h_biases[i][j] = 0.5f;
            #else
            h_biases[i][j] = d(gen);
            #endif
        }
    }

    d_weights = (float **)malloc(sizeof(float *) * n);
    d_biases = (float **)malloc(sizeof(float *) * n);

    for(int i = 1; i < n; i++){
        CHECK(cudaMalloc((void **)&d_weights[i], sizeof(float) * layers[i] * layers[i-1]));
        CHECK(cudaMalloc((void **)&d_biases[i], sizeof(float) * layers[i]));
    }

    for(int i = 1; i < n; i++){   
        cudaMemcpy(d_weights[i], h_weights[i], sizeof(float) * layers[i] * layers[i-1], cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases[i], h_biases[i], sizeof(float) * layers[i], cudaMemcpyHostToDevice);
    }

    float **activation;
    float **z;
    float **delta;
    float **nabla_w;
    float **nabla_b;

    // for debug
    float **nabla_w2;
    float **nabla_b2;
    nabla_b2 = (float **)malloc(sizeof(float *) * n);
    nabla_w2 = (float **)malloc(sizeof(float *) * n);
    for(int i = 1; i < n; i++){
        nabla_w2[i] = (float *)malloc(sizeof(float) * layers[i] * layers[i-1]);
        nabla_b2[i] = (float *)malloc(sizeof(float) * layers[i]);
    }

    activation = (float **)malloc(sizeof(float *) * n);
    z = (float **)malloc(sizeof(float *) * n);
    delta = (float **)malloc(sizeof(float *) * n);
    nabla_w = (float **)malloc(sizeof(float *) * n);
    nabla_b = (float **)malloc(sizeof(float *) * n);

    for(int i = 0; i < n; i++){
        cudaMalloc((void **)&activation[i], sizeof(float) * layers[i]);
        if(i){
            cudaMalloc((void **)&z[i], sizeof(float) * layers[i]);
            cudaMalloc((void **)&delta[i], sizeof(float) * layers[i]);
            cudaMalloc((void **)&nabla_w[i], sizeof(float) * layers[i] * layers[i-1]);
            cudaMalloc((void **)&nabla_b[i], sizeof(float) * layers[i]);
        }
    }

    // training
    int batch_size = 10;
    int num_batch = cnt1 / batch_size;
    int num_epoch = 30;
    float eta = 3.0f;

    #ifdef DEBUG
        num_epoch = 1;
        num_batch = 1;
        eta = 0.0f;
        
        float *first_output = (float *)malloc(sizeof(float) * layers[1]);
    #endif

    float *d_target;
    cudaMalloc((void **)&d_target, sizeof(float) * 10);
    
    float **batch_nabla_w;
    float **batch_nabla_b;

    batch_nabla_w = (float **)malloc(sizeof(float *) * n);
    batch_nabla_b = (float **)malloc(sizeof(float *) * n);

    for(int i = 1; i < n; i++){
        cudaMalloc((void **)&batch_nabla_w[i], sizeof(float) * layers[i] * layers[i-1]);
        cudaMalloc((void **)&batch_nabla_b[i], sizeof(float) * layers[i]);
    }

    for(int ep = 1; ep <= num_epoch; ep++){
        // shuffle data
        #ifndef DEBUG
        for(int i = 0; i < cnt1; i++){
            int j = rand() % cnt1;
            float *tmp = training_data[i];
            training_data[i] = training_data[j];
            training_data[j] = tmp;

            tmp = target[i];
            target[i] = target[j];
            target[j] = tmp;
        }
        #endif

        #define PBSTR "|||||||||||||||||||||||||||||||||||||||||||||||||"
        #define PBWIDTH 50

        // mini-batch SGD
        for(int i = 0; i < num_batch; i++){
            double progress = (double)i / num_batch;
            int val = (int)(progress * 100);
            int lpad = (int)(progress * PBWIDTH);
            int rpad = PBWIDTH - lpad;
            printf("\033[1m\033[34m\r%3d%% Epoch %d[%.*s%*s]\033[0m", val, ep, lpad, PBSTR, rpad, "");
            fflush(stdout);

            for(int j = 1; j < n; j++){
                cudaMemset(batch_nabla_w[j], 0, sizeof(float) * layers[j] * layers[j-1]);
                cudaMemset(batch_nabla_b[j], 0, sizeof(float) * layers[j]);
            }

            for(int j = 0; j < batch_size; j++){

                // init activation[0]
                // cudaMemcpy(activation[0], d_training_data[i], sizeof(float) * 784, cudaMemcpyDevic?eToDevice);
                // cudaMemcpy(d_target, target[i], sizeof(float) * 10, cudaMemcpyHostToDevice);
                int idx = i * batch_size + j;
                cudaMemcpy(activation[0], training_data[idx], sizeof(float) * 784, cudaMemcpyHostToDevice);
                cudaMemcpy(d_target, target[idx], sizeof(float) * 10, cudaMemcpyHostToDevice);

                // feedforward
                for(int l = 1; l < n; l++){
                    // kernel
                    // z = w * activation + b
                    // activation = sigmoid(z)
                    forward<<<1, layers[l]>>>(activation[l-1], d_weights[l], z[l], d_biases[l], activation[l], layers[l-1], layers[l]);
                }


                #ifdef DEBUG
                cudaMemcpy(first_output, activation[1], sizeof(float) * layers[1], cudaMemcpyDeviceToHost);
                for (int i = 0; i < layers[1]; i++)
                {
                    printf("%f ", first_output[i]);
                }
                
                #endif
                // backpropagation
                vecSub<<<1, layers[n-1]>>>(activation[n-1], d_target, delta[n-1], layers[n-1]);
                sigmoid_prime_vec<<<1, layers[n-1]>>>(activation[n-1], layers[n-1]);
                vecMul<<<1, layers[n-1]>>>(delta[n-1], z[n-1], delta[n-1], layers[n-1]); 
                

                get_nabla_b<<<1, layers[n-1]>>>(delta[n-1], nabla_b[n-1], layers[n-1]);
                get_nabla_w<<<1, layers[n-1]>>>(delta[n-1], activation[n-2], nabla_w[n-1], layers[n-1], layers[n-2]);
                

                for(int l = n-2; l > 0; l--){
                    // kernel
                    // delta = (w^T * delta) * sigmoid_prime(z)
                    // nabla_b = delta
                    // nabla_w = delta * activation^T
                    backprpoError<<<1, layers[l]>>>(delta[l], d_weights[l+1], delta[l+1], z[l], layers[l], layers[l+1]);
                    
                    get_nabla_b<<<1, layers[l]>>>(delta[l], nabla_b[l], layers[l]);
                    
                    get_nabla_w<<<1, layers[l]*layers[l-1]>>>(delta[l], activation[l-1], nabla_w[l], layers[l], layers[l-1]);    

                }
                    // print<<<1,1>>>(activation[n-1], 1);


                // update batch_nabla_w and batch_nabla_b
                for(int l = 1; l < n; l++){
                    vecAdd2<<<layers[l-1], layers[l]>>>(batch_nabla_w[l], nabla_w[l], layers[l] * layers[l-1]);
                    vecAdd2<<<1, layers[l]>>>(batch_nabla_b[l], nabla_b[l], layers[l]);  
                }
                // exit(0);
            }            
            
            for(int l = 1; l < n; l++){
                vecScale<<<1, layers[l]>>>(batch_nabla_w[l], eta / (float)batch_size, layers[l] * layers[l-1]);
                vecScale<<<1, layers[l]>>>(batch_nabla_b[l], eta / (float)batch_size, layers[l]);

                vecSub2<<<layers[l], layers[l-1]>>>(d_weights[l], batch_nabla_w[l], layers[l-1] * layers[l]);  
                vecSub2<<<1, layers[l]>>>(d_biases[l], batch_nabla_b[l], layers[l]);
            }
            
            // print<<<1, 10>>>(d_biases[2], 10);
            // print<<<1, 30>>>(d_weights[2], 30);

        }

        // evaluate kernel
        int correct = 0;
        for(int i = 0; i < cnt2; i++){
            cudaMemcpy(activation[0], test_data[i], sizeof(float) * 784, cudaMemcpyHostToDevice);
            for(int l = 1; l < n; l++){
                // kernel
                // z = w * activation + b
                // activation = sigmoid(z)
                forward<<<1, layers[l]>>>(activation[l-1], d_weights[l], d_biases[l], z[l], activation[l], layers[l-1], layers[l]);
            }

            float *output = (float *)malloc(sizeof(float) * layers[n-1]);
            cudaMemcpy(output, activation[n-1], sizeof(float) * layers[n-1], cudaMemcpyDeviceToHost);
            int max_idx = 0;
            for(int j = 1; j < layers[n-1]; j++){
                if(output[j] > output[max_idx]){
                    max_idx = j;
                }
            }
            if(max_idx == test_label[i]){
                correct++;
            }
        }
        printf("\033[1m Epoch %d: %d / %d\033[0m\n", ep, correct, cnt2);
        // exit(0);
    }



    // free memory
    for(int i = 1; i < n; i++){
        cudaFree(activation[i]);
        if(i){
        cudaFree(z[i]);
            cudaFree(delta[i]);
            cudaFree(nabla_w[i]);
            cudaFree(nabla_b[i]);
        }
    }

    cudaFree(activation);
    cudaFree(z);
    cudaFree(delta);
    cudaFree(nabla_w);
    cudaFree(nabla_b);

    for(int i = 1; i < n; i++){
        cudaFree(d_weights[i]);
        cudaFree(d_biases[i]);
    }

    cudaFree(d_weights);
    cudaFree(d_biases);

    for(int i = 0; i < cnt1; i++){
        free(training_data[i]);
        free(target[i]);
    }

    free(training_data);
    free(target);

    for(int i = 1; i < n; i++){
        free(h_weights[i]);
        free(h_biases[i]);
    }

    free(h_weights);
    free(h_biases);

    free(data1);
    free(data2);

    for(int i = 0; i < cnt2; i++){
        free(test_data[i]);
    }
    free(test_data);
    
    free(test_label);

    for(int i = 1; i < n; i++){
        free(nabla_w2[i]);
        free(nabla_b2[i]);
        cudaFree(batch_nabla_w[i]);
        cudaFree(batch_nabla_b[i]);
    }
    free(nabla_w2);
    free(nabla_b2);

    cudaFree(batch_nabla_w);
    cudaFree(batch_nabla_b);

    return 0;
}



