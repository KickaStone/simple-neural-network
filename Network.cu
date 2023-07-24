#include "network.cuh"
void NeuralNetwork::fillRandom(double *arr, int size)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, clock());
    // curandGenerateNormal(gen, arr, size, 0, 1); // standard normal distribution
    curandGenerateNormalDouble(gen, arr, size, 0, 1); // standard normal distribution
    CUDA_CHECK(cudaGetLastError());
}

void NeuralNetwork::fillZero(double *arr, int size)
{
    CUDA_CHECK(cudaMemset(arr, 0, sizeof(double) * size));
}

NeuralNetwork::NeuralNetwork(int input, std::vector<int> layers, double eta)
{
    spdlog::debug("Initializing network");
    params.inputsize = input;
    params.outputsize = layers[layers.size() - 1];
    params.num_layers = layers.size();
    params.layers = layers;
    params.eta = eta;

    CUDA_CHECK(cudaMalloc((void**)&data, sizeof(double) * input));
    CUDA_CHECK(cudaMalloc((void**)&y, sizeof(double) * layers[layers.size() - 1]));

    int blocks = 0;
    double *dev_w, *dev_b, *dev_z, *dev_a, *dev_dz, *dev_dC_dw, *dev_db, *dev_da;
    layers.insert(layers.begin(), input);
    for(int i = 0; i < params.num_layers; i++){
        // note i include input layer
        // create and initialize weights and biases
        CUDA_CHECK(cudaMalloc((void**)&dev_w, sizeof(double) * layers[i+1] * layers[i]));
        CUDA_CHECK(cudaMalloc((void**)&dev_b, sizeof(double) * layers[i+1]));
        fillRandom(dev_w, layers[i+1] * layers[i]);
        fillRandom(dev_b, layers[i+1]);
        // blocks = (layers[i+1] + params.blocksize - 1) / params.blocksize;
        // memset<<<blocks, params.blocksize>>>(dev_b, 0.0f, layers[i+1]);
        CUDA_CHECK(cudaGetLastError());
        w.push_back(dev_w);
        b.push_back(dev_b);

        // initialize z and a
        CUDA_CHECK(cudaMalloc((void**)&dev_z, sizeof(double) * layers[i+1]));
        CUDA_CHECK(cudaMalloc((void**)&dev_a, sizeof(double) * layers[i+1]));
        z.push_back(dev_z);
        a.push_back(dev_a);

        // initialize grad vectors
        CUDA_CHECK(cudaMalloc((void**)&dev_dC_dw, sizeof(double) * layers[i+1] * layers[i]));
        CUDA_CHECK(cudaMalloc((void**)&dev_db, sizeof(double) * layers[i+1]));
        // fillRandom(dev_dC_dw, layers[i+1] * layers[i]);
        // memset<<<blocks, params.blocksize>>>(dev_db, 0.1f, layers[i+1]);
        dC_dw.push_back(dev_dC_dw);
        dC_db.push_back(dev_db);

        // initialize dz da
        CUDA_CHECK(cudaMalloc((void**)&dev_dz, sizeof(double) * layers[i+1]));
        CUDA_CHECK(cudaMalloc((void**)&dev_da, sizeof(double) * layers[i+1]));
        dC_dz.push_back(dev_dz);
        dC_da.push_back(dev_da);

        // initialize z_prime
        CUDA_CHECK(cudaMalloc((void**)&dev_dz, sizeof(double) * layers[i+1]));
        z_prime.push_back(dev_dz);

        // initialize batch_dw batch_db
        CUDA_CHECK(cudaMalloc((void**)&dev_dC_dw, sizeof(double) * layers[i+1] * layers[i]));
        CUDA_CHECK(cudaMalloc((void**)&dev_db, sizeof(double) * layers[i+1]));
        batch_dw.push_back(dev_dC_dw);
        batch_db.push_back(dev_db);
    }
    spdlog::debug("Network initialized");
}

NeuralNetwork::~NeuralNetwork()
{
    for(int i = 0; i < params.num_layers; i++){
        CUDA_CHECK(cudaFree(w[i]));
        CUDA_CHECK(cudaFree(b[i]));
        CUDA_CHECK(cudaFree(z[i]));
        CUDA_CHECK(cudaFree(a[i]));
        CUDA_CHECK(cudaFree(dC_dw[i]));
        CUDA_CHECK(cudaFree(dC_db[i]));
        CUDA_CHECK(cudaFree(dC_dz[i]));
        CUDA_CHECK(cudaFree(dC_da[i]));
        CUDA_CHECK(cudaFree(z_prime[i]));
    }
}


double* NeuralNetwork::forward(double *input, int size)
{
    spdlog::debug("Forward propagation");
    double *output = new double[params.outputsize]();
    if(size != params.inputsize){
        std::cout << "Input size does not match network input size" << std::endl;
        return NULL;
    }

    // copy input data to first layer
    CUDA_CHECK(cudaMemcpy(data, input, sizeof(double) * size, cudaMemcpyHostToDevice));
    
    for(int l = 0; l < params.num_layers; l++){
        spdlog::debug("Layer {}", l);
        blocks = std::ceil((params.layers[l] + params.blocksize - 1)/ params.blocksize);
        CUDA_CHECK(cudaMemset(z[l], 0, params.layers[l] * sizeof(double)));
        CUDA_CHECK(cudaMemset(a[l], 0, params.layers[l] * sizeof(double)));

        if(l == 0) matMulvec<<<blocks, params.blocksize>>>(w[l], data, z[l], params.layers[l], params.inputsize);
        else matMulvec<<<blocks, params.blocksize>>>(w[l], a[l-1], z[l], params.layers[l], params.layers[l-1]);
        CUDA_CHECK(cudaGetLastError());

        // add bias
        vecAdd<<<blocks, params.blocksize>>>(z[l], b[l], z[l], params.layers[l]);
        CUDA_CHECK(cudaGetLastError());

        // apply sigmoid
        sigmoid_ztoa<<<blocks, params.blocksize>>>(z[l], a[l], params.layers[l]);
    }

    // copy output to host
    CUDA_CHECK(cudaMemcpy(output, a[params.num_layers - 1], sizeof(double) * params.outputsize, cudaMemcpyDeviceToHost));
    spdlog::debug("Forward propagation done");
    return output;
}

void NeuralNetwork::setParams(double learning_rate, int batch_size)
{
    params.eta = learning_rate;
    params.batchsize = batch_size;
}

void NeuralNetwork::backprop(double *h_y)
{
    spdlog::debug("Backpropagation");
    int n = -1; // n  number of neurons in layer l
    // copy y to device
    CUDA_CHECK(cudaMemcpy(this->y, h_y, sizeof(double) * params.outputsize, cudaMemcpyHostToDevice));
    for(int l = params.num_layers-1; l >= 0; l--){
        spdlog::debug("Layer {}", l);

        n = params.layers[l];

        blocks = std::ceil((n + params.blocksize - 1)/ params.blocksize);

        // calculate z_prime
        sigmoid_z_prime<<<blocks, params.blocksize>>>(a[l], z_prime[l], n);
        CUDA_CHECK(cudaGetLastError());

        // calculate delta
        if(l == params.num_layers-1){
            cost_prime<<<blocks, params.blocksize>>>(a[l], y, dC_da[l], n); // dC_da = a - y
        }else{
            // dC_da = w[l+1]^T * dC_dz[l+1]
            // note: w[L+1]'s size is (n[l+1] * n[l])
            matMulvec<<<blocks, params.blocksize>>>(w[l+1], dC_dz[l+1], dC_da[l], params.layers[l+1], params.layers[l], true); // dC_da = w[l+1]^T * dC_dz[l+1]
        }
        
        CUDA_CHECK(cudaGetLastError());
        vecMul<<<blocks, params.blocksize>>>(dC_da[l], z_prime[l], dC_dz[l], n); // dC_dz = dC_da * z_prime
        CUDA_CHECK(cudaGetLastError());

        spdlog::debug("calculate dC_dw and dC_db");

        // calculate dC_db
        copy<<<blocks, params.blocksize>>>(dC_db[l], dC_dz[l], n);
        CUDA_CHECK(cudaGetLastError());
        update<<<blocks, params.blocksize>>>(batch_db[l], dC_db[l], -1.0f, n);
        CUDA_CHECK(cudaGetLastError());

        // calculate dC_dw = dC_dz * a[l-1]
        if(l != 0){
            blocks = std::ceil((n * params.layers[l-1] + params.blocksize - 1)/ params.blocksize);
            cal_dw<<<blocks, params.blocksize>>>(a[l-1], dC_dz[l], dC_dw[l], params.layers[l-1], params.layers[l]);
            CUDA_CHECK(cudaGetLastError());
            update<<<blocks, params.blocksize>>>(batch_dw[l], dC_dw[l], -1.0f, n * params.layers[l-1]);
        }else{
            blocks = std::ceil((n * params.inputsize + params.blocksize - 1)/ params.blocksize);
            cal_dw<<<blocks, params.blocksize>>>(data, dC_dz[l], dC_dw[l], params.inputsize, params.layers[l]);
            CUDA_CHECK(cudaGetLastError());
            update<<<blocks, params.blocksize>>>(batch_dw[l], dC_dw[l], -1.0f, n * params.inputsize);
        }
        
        CUDA_CHECK(cudaGetLastError());
    }
    spdlog::debug("Backpropagation done");
}

void NeuralNetwork::update_weights_and_biases()
{
    // update weights
    spdlog::debug("Updating weights and biases");
    int n = params.inputsize;
    for(int l = 0; l < params.num_layers; l++){
        blocks = std::ceil((params.layers[l] * n + params.blocksize - 1)/ params.blocksize);
        update<<<blocks, params.blocksize>>>(w[l], batch_dw[l], params.eta/params.batchsize, params.layers[l] * n);
        CUDA_CHECK(cudaGetLastError());
        n = params.layers[l];
        blocks = std::ceil((params.layers[l] + params.blocksize - 1)/ params.blocksize);
        update<<<blocks, params.blocksize>>>(b[l], batch_db[l], params.eta/params.batchsize, params.layers[l]);
        CUDA_CHECK(cudaGetLastError());
    }
    spdlog::debug("Weights and biases updated");
}

void NeuralNetwork::SDG_train(std::vector<double *> &training_data, std::vector<double *> training_label, int epochs, int batch_size, std::vector<double *> &test_data, std::vector<int> &test_label)
{
    params.batchsize = batch_size;
    for(int ep = 0; ep < epochs; ep++){

        CUDA_CHECK(cudaDeviceSynchronize());
        spdlog::debug("Epoch {}", ep);
        void *d_data, *d_label;
        loss = 0.0f;

        srand(time(NULL));
        // shuffle training data
        for(int i = 0; i < training_data.size(); i++){
            int j = rand() % training_data.size();
            std::swap(training_data[i], training_data[j]);
            std::swap(training_label[i], training_label[j]);
        }

        // iterate through batches
        for(int batch = 0; batch < training_data.size() / batch_size; batch++){
            spdlog::debug("Batch {}", batch);
            #define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||"
            #define PBWIDTH 50
            double percentage = (double)(batch+1) / (double)(training_data.size() / batch_size);
            int lpad = (int)(percentage * PBWIDTH);
            int val = (int)(percentage * 100);
            int rpad = PBWIDTH - lpad;
            printf("\033[1m\033[34m\r%3d%% Epoch %02d, Batch %05d[%.*s%*s]\033[0m", val, ep, batch+1, lpad, PBSTR, rpad, "");
            fflush(stdout);
            mini_batch(training_data, training_label, batch_size, batch * batch_size);
        }

        fflush(stdout);
        int correct = 0;
        double test_loss = 0.0f;
        for(int i = 0; i < test_data.size(); i++){
            double *output = forward(test_data[i], params.inputsize);
            int max_index = 0;
            for(int j = 0; j < params.outputsize; j++){
                if(output[j] > output[max_index]) max_index = j;
                if(j == test_label[i]) test_loss +=  (1.0f - output[j]) * (1.0f - output[j]);
                else test_loss +=  output[j] * output[j];
            }
            if(max_index == test_label[i]) correct++;
        }
        test_loss /= 2;
        
        printf("%d/%d, Train loss: %lf, Test loss: %lf\n", correct, test_data.size(), loss, test_loss);
    }
}

void NeuralNetwork::mini_batch(std::vector<double *> &training_data, std::vector<double*> &training_label, int batch_size, int start)
{
    spdlog::debug("Mini batch");
    int end = start + batch_size;
    if(end > training_data.size()){
        std::cout << "Invalid batch size" << std::endl;
        return;
    }

    // initialize batch_dw and batch_db
    int n = params.inputsize;
    for(int l = 0; l < params.num_layers; l++){
        CUDA_CHECK(cudaMemset(batch_dw[l], 0, sizeof(double) * n * params.layers[l]));
        CUDA_CHECK(cudaMemset(batch_db[l], 0, sizeof(double) * params.layers[l]));
        n = params.layers[l];
    }

    for(int i = start; i < end; i++){
        auto output = forward(training_data[i], params.inputsize);
        for(int j = 0; j < params.outputsize; j++){
            loss +=0.5f * (training_label[i][j] - output[j]) * (training_label[i][j] - output[j]);
        }
        backprop(training_label[i]);
    }

    // update weights and biases    
    update_weights_and_biases();
    spdlog::debug("Mini batch done");

    CUDA_CHECK(cudaDeviceSynchronize());
}



#ifdef DEBUG
void NeuralNetwork::_debug_get_weights_and_biases(std::vector<double *> w, std::vector<double *> b)
{
    // copy weights and biases to host
    spdlog::debug("Getting weights and biases");
    int n = params.inputsize;
    for(int i = 0; i < params.num_layers; i++){
        CUDA_CHECK(cudaMemcpy(w[i], this->w[i], sizeof(double) * n * params.layers[i], cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b[i], this->b[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
        n = params.layers[i];
    }
    spdlog::debug("Weights and biases got");
}

void NeuralNetwork::_debug_get_grad(std::vector<double *> dw, std::vector<double *> db)
{
    // copy weights and biases to host
    spdlog::debug("Getting gradients");
    int n = params.inputsize;
    for(int i = 0; i < params.num_layers; i++){
        CUDA_CHECK(cudaMemcpy(dw[i], this->dC_dw[i], sizeof(double) * n * params.layers[i], cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(db[i], this->dC_db[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
        n = params.layers[i];
    }
    spdlog::debug("Gradients got");
}

Params NeuralNetwork::_debug_params()
{
    return params;
}

void NeuralNetwork::_debug_get_a(std::vector<double *> a)
{
    // copy a values to host
    spdlog::debug("Getting activations values");
    
    for(int i = 0; i < params.num_layers; i++){
        CUDA_CHECK(cudaMemcpy(a[i], this->a[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
    }
    spdlog::debug("Activations values got");
}

void NeuralNetwork::_debug_get_delta(std::vector<double *> delta)
{
    // copy a values to host
    spdlog::debug("Getting delta values");
    
    for(int i = 0; i < params.num_layers; i++){
        CUDA_CHECK(cudaMemcpy(delta[i], this->dC_dz[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
    }
    spdlog::debug("Delta values got");
}

void NeuralNetwork::_debug_set(std::vector<double *> w, std::vector<double *> b)
{  
    spdlog::debug("Setting weights and biases");
    if(w.size() != params.num_layers || b.size() != params.num_layers){
        std::cout << "Invalid weight or bias vector size" << std::endl;
        return;
    }

    // copy weights and biases to device
    int n = params.inputsize;
    for(int i = 0; i < params.num_layers; i++){
        CUDA_CHECK(cudaMemcpy(this->w[i], w[i], sizeof(double) * n * params.layers[i], cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->b[i], b[i], sizeof(double) * params.layers[i], cudaMemcpyHostToDevice));
        n = params.layers[i];
    }
    spdlog::debug("Weights and biases set");
}
#endif