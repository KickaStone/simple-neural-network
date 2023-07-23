#include <stdio.h>
#include <cuda_runtime.h>

#include "Mnist_helper.h"
#include <random>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <time.h>

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

    #ifndef DEBUG
    // load mnist data
    int cnt1 = 60000;
    int cnt2 = 10000;
    
    float **training_data = (float **)malloc(sizeof(float *) * cnt1);
    int *training_label = (int *)malloc(sizeof(int) * cnt1);
    float **test_data = (float **)malloc(sizeof(float *) * cnt2);
    int *test_label = (int *)malloc(sizeof(int) * cnt2);

    for (int i = 0; i < cnt1; i++) training_data[i] = (float *)malloc(sizeof(float) * 784);
    for (int i = 0; i < cnt2; i++) test_data[i] = (float *)malloc(sizeof(float) * 784);
    
    load(training_data, training_label, test_data, test_label);
    // create network

    int n = 3;
    int *layers = (int *)malloc(sizeof(int) * n);
    layers[0] = 784;
    layers[1] = 30;
    layers[2] = 10;

    #else

    int cnt1 = 1;
    int cnt2 = 0;
    float **training_data = (float **)malloc(sizeof(float *) * cnt1);
    float **target = (float **)malloc(sizeof(float *) * cnt1);
    float **test_data = (float **)malloc(sizeof(float *) * cnt2);
    int *test_label = (int *)malloc(sizeof(int) * cnt2);

    for (int i = 0; i < cnt1; i++)
    {
        training_data[i] = (float *)malloc(sizeof(float) * 2);
        target[i] = (float *)malloc(sizeof(float) * 2);
    }

    training_data[0][0] = 0.05f;
    training_data[0][1] = 0.1f;
    target[0][0] = 0.01f;
    target[0][1] = 0.99f;


    int n = 3;
    int *layers = (int *)malloc(sizeof(int) * n);
    layers[0] = 2;
    layers[1] = 2;
    layers[2] = 2;
    #endif

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

    #ifdef DEBUG

    h_weights[1][0] = 0.15f;
    h_weights[1][1] = 0.2f;
    h_weights[1][2] = 0.25f;
    h_weights[1][3] = 0.3f;
    h_weights[2][0] = 0.4f;
    h_weights[2][1] = 0.45f;
    h_weights[2][2] = 0.5f;
    h_weights[2][3] = 0.55f;

    h_biases[1][0] = 0.35f;
    h_biases[1][1] = 0.35f;
    h_biases[2][0] = 0.6f; 
    h_biases[2][1] = 0.6f;
    #else

    // initialize weights and biases

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    
    for(int i = 1; i < n; i++){
        for(int j = 0; j < layers[i]; j++){
            for(int k = 0; k < layers[i-1]; k++){
                h_weights[i][j * layers[i-1] + k] = distribution(generator);
            }
            h_biases[i][j] = distribution(generator);
        }
    }

    #endif

    d_weights = (float **)malloc(sizeof(float *) * n);
    d_biases = (float **)malloc(sizeof(float *) * n);

    for(int i = 1; i < n; i++){
        CHECK(cudaMalloc((void **)&d_weights[i], sizeof(float) * layers[i] * layers[i-1]));
        CHECK(cudaMalloc((void **)&d_biases[i], sizeof(float) * layers[i])); 
        cudaMemcpy(d_weights[i], h_weights[i], sizeof(float) * layers[i] * layers[i-1], cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases[i], h_biases[i], sizeof(float) * layers[i], cudaMemcpyHostToDevice);
    }

    float **activation;
    float **z;
    float **delta;
    float **nabla_w;
    float **nabla_b;

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
        batch_size = 1;
        num_epoch = 1;
        num_batch = 1;
        eta = 0.5f;
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
    int *d_label;
    cudaMalloc((void **)&d_label, sizeof(int) * cnt1);

    // training start
    for(int ep = 0; ep < num_epoch; ep++){
        // shuffle data
        #ifndef DEBUG
        srand(time(NULL));
        for(int i = 0; i < cnt1; i++){
            int j = rand() % cnt1;
            float *tmp = training_data[i];
            training_data[i] = training_data[j];
            training_data[j] = tmp;
            
            int tmp2 = training_label[i];
            training_label[i] = training_label[j];
            training_label[j] = tmp2;
        }

        cudaMemcpy(d_label, training_label, sizeof(int) * cnt1, cudaMemcpyHostToDevice);

        // copy train data to device
        #endif

        #define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||"
        #define PBWIDTH 50

        double total_cost = 0.0f;

        // mini-batch SGD

        for(int i = 0; i < num_batch; i++){
            double progress = (double)(i+1) / num_batch;
            int val = (int)(progress * 100);
            int lpad = (int)(progress * PBWIDTH);
            int rpad = PBWIDTH - lpad;
            printf("\033[1m\033[34m\r%3d%% Epoch %02d, Batch %05d[%.*s%*s]\033[0m", val, ep, i, lpad, PBSTR, rpad, "");
            fflush(stdout);

            for(int j = 1; j < n; j++){
                cudaMemset(batch_nabla_w[j], 0, sizeof(float) * layers[j] * layers[j-1]);
                cudaMemset(batch_nabla_b[j], 0, sizeof(float) * layers[j]);
            }

            // mini-batch start
            for(int j = 0; j < batch_size; j++){
                int idx = i * batch_size + j;
            #ifdef DEBUG
                cudaMemcpy(activation[0], training_data[0], sizeof(float) * 2, cudaMemcpyHostToDevice);
                cudaMemcpy(d_target, target[0], sizeof(float) * 2, cudaMemcpyHostToDevice);
            #else
                cudaMemcpy(activation[0], training_data[idx], sizeof(float) * layers[0], cudaMemcpyHostToDevice);

                printf("%d\n", training_label[idx]);
                float *img = (float *)malloc(sizeof(float) * 784);
                cudaMemcpy(img, activation[0], sizeof(float) * 784, cudaMemcpyDeviceToHost);
                for(int k = 0; k < 28; k++){
                    for(int l = 0; l < 28; l++){
                        if(img[k * 28 + l] > 0.1f){
                            printf("\033[1m\033[31m%d\033[0m", 1);
                        }else{
                            printf("%d", 0);
                        }
                    }
                    printf("\n");
                }
                free(img);
                getchar();
            #endif
                // feedforward
                for(int l = 1; l < n; l++)
                    forward<<<1, layers[l]>>>(activation[l-1], d_weights[l], z[l], d_biases[l], activation[l], layers[l-1], layers[l]);

                float *result = (float *)malloc(sizeof(float) * layers[n-1]);
                cudaMemcpy(result, activation[n-1], sizeof(float) * layers[n-1], cudaMemcpyDeviceToHost);
                float cost = 0.0f;
                for(int l = 0; l < layers[n-1]; l++){
                    if(l==training_label[idx]){
                        cost += 0.5f * (result[l] - 1.0f) * (result[l] - 1.0f);
                    }else{
                        cost += 0.5f * result[l] * result[l];
                    }
                }
                free(result);
                total_cost += cost;

                // backpropagation
                cost_derivative<<<1, layers[n-1]>>>(activation[n-1], training_label[idx], delta[n-1], layers[n-1]);
                sigmoid_prime_vec<<<1, layers[n-1]>>>(z[n-1], layers[n-1]);
                vecMul<<<1, layers[n-1]>>>(delta[n-1], z[n-1], delta[n-1], layers[n-1]);
                
                get_nabla_b<<<1, layers[n-1]>>>(delta[n-1], nabla_b[n-1], layers[n-1]);
                get_nabla_w<<<layers[n-1], layers[n-2]>>>(delta[n-1], activation[n-2], nabla_w[n-1], layers[n-1], layers[n-2]);

                for(int l = n-2; l > 0; l--){
                    backprpoError<<<1, layers[l]>>>(delta[l], d_weights[l+1], delta[l+1], activation[l], layers[l], layers[l+1]);
                    get_nabla_b<<<1, layers[l]>>>(delta[l], nabla_b[l], layers[l]);
                    get_nabla_w<<<layers[l], layers[l-1]>>>(delta[l], activation[l-1], nabla_w[l], layers[l], layers[l-1]);    
                }

                // update batch_nabla_w and batch_nabla_b
                for(int l = 1; l < n; l++){
                    vecAdd2<<<layers[l-1], layers[l]>>>(batch_nabla_w[l], nabla_w[l], layers[l] * layers[l-1]);
                    vecAdd2<<<1, layers[l]>>>(batch_nabla_b[l], nabla_b[l], layers[l]);  
                }
            }            
            
            for(int l = 1; l < n; l++){
                vecScale<<<layers[l], layers[l-1]>>>(batch_nabla_w[l], eta / (float)batch_size, layers[l] * layers[l-1]);
                vecScale<<<1, layers[l]>>>(batch_nabla_b[l], eta / (float)batch_size, layers[l]);
                vecSub2<<<layers[l], layers[l-1]>>>(d_weights[l], batch_nabla_w[l], layers[l-1] * layers[l]);  
                vecSub2<<<1, layers[l]>>>(d_biases[l], batch_nabla_b[l], layers[l]);
            }
        }

        // evaluate kernel
        int correct = 0;
        float test_cost = 0.0f;
        for(int i = 0; i < cnt2; i++){
            cudaMemcpy(activation[0], test_data[i], sizeof(float) * 784, cudaMemcpyHostToDevice);
            for(int l = 1; l < n; l++){
                forward<<<1, layers[l]>>>(activation[l-1], d_weights[l], d_biases[l], z[l], activation[l], layers[l-1], layers[l]);
            }

            float *output = (float *)malloc(sizeof(float) * layers[n-1]);
            cudaMemcpy(output, activation[n-1], sizeof(float) * layers[n-1], cudaMemcpyDeviceToHost);
            int max_idx = 0;
            for(int j = 0; j < layers[n-1]; j++){
                if(output[j] > output[max_idx]){
                    max_idx = j;
                }
                if(test_label[i] == j){
                    test_cost += 0.5f * (output[j] - 1.0f) * (output[j] - 1.0f);
                }else{
                    test_cost += 0.5f * output[j] * output[j];
                }

            }
            if(max_idx == test_label[i]){
                correct++;
            }
        }
        printf("\033[1m Epoch %d: %d / %d, total_cost=%lf, test_cost=%lf\033[0m\n", ep, correct, cnt2, total_cost, test_cost);
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
    }

    free(training_data);

    for(int i = 1; i < n; i++){
        free(h_weights[i]);
        free(h_biases[i]);
    }

    free(h_weights);
    free(h_biases);

    #ifndef DEBUG
    #endif

    free(training_label);
    free(test_data);
    free(test_label);

    for(int i = 1; i < n; i++){
        cudaFree(batch_nabla_w[i]);
        cudaFree(batch_nabla_b[i]);
    }

    cudaFree(batch_nabla_w);
    cudaFree(batch_nabla_b);

    return 0;
}



