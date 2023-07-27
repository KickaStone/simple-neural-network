#include "network.cuh"
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 50

const double ALPHA = 1.0;
const double Beta = 0.0;
const double ALPHA_NEG = -1.0;


// ===================== kernels =======================
__global__ void hadamard_product(double *a, double *b, double *c, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] * b[i];
}

__global__ void sigmoid(double *z, double *a, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] = 1.0 / (1.0 + exp(-z[i]));
}

__global__ void sigmoid_derivative(double *a, double *z, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        z[i] = a[i] * (1.0f - a[i]);
}
// ======================================================

void NeuralNetwork::fillRandom(double *arr, int size)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, clock());
    curandGenerateNormalDouble(gen, arr, size, 0, 1); // standard normal distribution
    CUDA_CHECK(cudaGetLastError());
}

void NeuralNetwork::fillZero(double *arr, int size)
{
    CUDA_CHECK(cudaMemset(arr, 0, sizeof(double) * size));
}

NeuralNetwork::NeuralNetwork(int input, std::vector<int> layers)
{

    params.inputsize  = input;
    params.outputsize = layers[layers.size() - 1];
    params.num_layers = layers.size();
    params.layers     = layers;
    params.eta        = 0;
    params.batchsize  = 0;

    CUBLAS_CHECK(cublasCreate(&this->cublasH));

    CUDA_CHECK(cudaMalloc((void **)&data, sizeof(double) * input));
    CUDA_CHECK(cudaMalloc((void **)&y, sizeof(double) * layers[layers.size() - 1]));

    double *dev_w, *dev_b, *dev_z, *dev_a, *dev_dz, *dev_dC_dw, *dev_db, *dev_da;
    layers.insert(layers.begin(), input);
    for (int i = 0; i < params.num_layers; i++)
    {
        // note i include input layer
        // create and initialize weights and biases
        CUDA_CHECK(cudaMalloc((void **)&dev_w, sizeof(double) * layers[i + 1] * layers[i]));
        CUDA_CHECK(cudaMalloc((void **)&dev_b, sizeof(double) * layers[i + 1]));
        fillRandom(dev_w, layers[i + 1] * layers[i]);
        CUDA_CHECK(cudaMemset(dev_b, 0, layers[i + 1] * sizeof(double)));
        CUDA_CHECK(cudaGetLastError());
        w.push_back(dev_w);
        b.push_back(dev_b);

        // initialize z and a
        CUDA_CHECK(cudaMalloc((void **)&dev_z, sizeof(double) * layers[i + 1]));
        CUDA_CHECK(cudaMalloc((void **)&dev_a, sizeof(double) * layers[i + 1]));
        z.push_back(dev_z);
        a.push_back(dev_a);

        // initialize grad vectors
        CUDA_CHECK(cudaMalloc((void **)&dev_dC_dw, sizeof(double) * layers[i + 1] * layers[i]));
        CUDA_CHECK(cudaMalloc((void **)&dev_db, sizeof(double) * layers[i + 1]));
        dC_dw.push_back(dev_dC_dw);
        dC_db.push_back(dev_db);

        // initialize dz da
        CUDA_CHECK(cudaMalloc((void **)&dev_dz, sizeof(double) * layers[i + 1]));
        CUDA_CHECK(cudaMalloc((void **)&dev_da, sizeof(double) * layers[i + 1]));
        dC_dz.push_back(dev_dz);
        dC_da.push_back(dev_da);

        // initialize z_prime
        CUDA_CHECK(cudaMalloc((void **)&dev_dz, sizeof(double) * layers[i + 1]));
        z_prime.push_back(dev_dz);
    }
}

NeuralNetwork::~NeuralNetwork()
{
    try{
        for (int i = 0; i < params.num_layers; i++)
        {
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
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(y));
        CUBLAS_CHECK(cublasDestroy(this->cublasH));
    }catch(...){
        std::cout << "Error in destructor" << std::endl;
        exit(1);
    }
}

void NeuralNetwork::setParams(double learning_rate, int batch_size, int epochs)
{
    params.eta       = learning_rate;
    params.batchsize = batch_size;
    params.epochs    = epochs;
}

double *NeuralNetwork::forward(double *input, int size)
{
    if (size != params.inputsize)
    {
        std::cout << "Input size does not match network input size" << std::endl;
        return NULL;
    }

    // copy input data to first layer
    CUDA_CHECK(cudaMemcpy(data, input, sizeof(double) * size, cudaMemcpyHostToDevice));

    for (int l = 0; l < params.num_layers; l++)
    {

        blocks = std::ceil((params.layers[l] + params.blocksize - 1) / params.blocksize);
        CUDA_CHECK(cudaMemset(z[l], 0, params.layers[l] * sizeof(double)));
        CUDA_CHECK(cudaMemset(a[l], 0, params.layers[l] * sizeof(double)));

        if (l == 0)
            CUBLAS_CHECK(
                cublasDgemv(
                    cublasH,
                    CUBLAS_OP_N, params.layers[l], params.inputsize,
                    &ALPHA, w[l], params.layers[l],
                    data, 1,
                    &Beta, z[l], 1
                )
            );
        else
            CUBLAS_CHECK(
                cublasDgemv(
                    cublasH,
                    CUBLAS_OP_N, params.layers[l], params.layers[l - 1],
                    &ALPHA, w[l], params.layers[l],
                    a[l - 1], 1,
                    &Beta, z[l], 1
                )
            );

        // add bias
        CUBLAS_CHECK(cublasDaxpy(cublasH, params.layers[l], &ALPHA, b[l], 1, z[l], 1));

        // apply sigmoid
        sigmoid<<<blocks, params.blocksize>>>(z[l], a[l], params.layers[l]);
        CUDA_CHECK(cudaGetLastError());
    }
    // copy output to host
    double *output = new double[params.outputsize]();
    CUDA_CHECK(cudaMemcpy(output, a[params.num_layers - 1], sizeof(double) * params.outputsize, cudaMemcpyDeviceToHost));
    return output;
}


void NeuralNetwork::backprop(double *h_y)
{
    
    int n = -1; // n  number of neurons in layer l
    // copy y to device
    CUDA_CHECK(cudaMemcpy(this->y, h_y, sizeof(double) * params.outputsize, cudaMemcpyHostToDevice));
    for (int l = params.num_layers - 1; l >= 0; l--)
    {
        
        n = params.layers[l];

        blocks = std::ceil((n + params.blocksize - 1) / params.blocksize);

        // calculate z_prime
        sigmoid_derivative<<<blocks, params.blocksize>>>(a[l], z_prime[l], n);
        CUDA_CHECK(cudaGetLastError());

        // calculate delta
        if (l == params.num_layers - 1)
        {
            CUBLAS_CHECK(
                cublasDgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, &ALPHA, a[l], n,&ALPHA_NEG, y, n, dC_da[l], n)
            );
        }
        else
        {
            // dC_da = w[l+1]^T * dC_dz[l+1]
            CUBLAS_CHECK(
                cublasDgemv(
                    cublasH,
                    CUBLAS_OP_T, params.layers[l + 1], params.layers[l],
                    &ALPHA, w[l + 1], params.layers[l + 1],
                    dC_dz[l + 1], 1,
                    &Beta, dC_da[l], 1
                )
            );
        }

        CUDA_CHECK(cudaGetLastError());
        hadamard_product<<<blocks, params.blocksize>>>(dC_da[l], z_prime[l], dC_dz[l], n); // dC_dz = dC_da * z_prime
        CUDA_CHECK(cudaGetLastError());

        // calculate dC_db
        double alpha_db = 1.0;
        CUBLAS_CHECK(cublasDaxpy(cublasH, n, &alpha_db, dC_dz[l], 1, dC_dw[l], 1));
        
        // calculate dC_dw = dC_dz * a[l-1]
        if (l != 0)
        {
            blocks = std::ceil((n * params.layers[l - 1] + params.blocksize - 1) / params.blocksize);
            CUBLAS_CHECK(
                cublasDgemm(
                    cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, params.layers[l - 1], 1,
                    &ALPHA, dC_dz[l], n, a[l - 1], 1,
                    &ALPHA, dC_dw[l], n
                )
            );
        }
        else
        {
            CUBLAS_CHECK(
                cublasDgemm(
                    cublasH,
                    CUBLAS_OP_N,CUBLAS_OP_N,
                    n, params.inputsize, 1,
                    &ALPHA, dC_dz[l], n, data, 1,
                    &ALPHA, dC_dw[l], n
                )
            );
        }
    }
}

void NeuralNetwork::update_weights_and_biases()
{
    double alpha = -1.0 * params.eta / params.batchsize;
    int n = params.inputsize;
    for (int l = 0; l < params.num_layers; l++)
    {
        CUBLAS_CHECK(cublasDaxpy(this->cublasH, n * params.layers[l], &alpha, dC_dw[l], 1, w[l], 1));
        CUBLAS_CHECK(cublasDaxpy(this->cublasH, params.layers[l], &alpha, dC_dw[l], 1, b[l], 1));
        n = params.layers[l];
    }
}

void NeuralNetwork::train(std::vector<double *> &training_data, std::vector<double *> training_label, std::vector<double *> &test_data, std::vector<int> &test_label)
{
    if (params.epochs == 0 || params.batchsize == 0 || params.eta == 0)
    {
        std::cout << "Invalid parameters! set params first!" << std::endl;
        return;
    }

    for (int ep = 1; ep <= params.epochs; ep++)
    {
        CUDA_CHECK(cudaDeviceSynchronize());

        loss = 0.0f;

        srand(time(NULL));
        // shuffle training data
        for (int i = 0; i < training_data.size(); i++)
        {
            int j = rand() % training_data.size();
            std::swap(training_data[i], training_data[j]);
            std::swap(training_label[i], training_label[j]);
        }

        // iterate through batches
        for (int batch = 0; batch < training_data.size() / params.batchsize; batch++)
        {

            double percentage = (double)(batch + 1) / (double)(training_data.size() / params.batchsize);
            int lpad = (int)(percentage * PBWIDTH);
            int val = (int)(percentage * 100);
            int rpad = PBWIDTH - lpad;
            printf("\033[1m\033[34m\r%3d%% Epoch %02d/%02d, Batch %05d[%.*s%*s]\033[0m", val, ep, params.epochs, batch + 1, lpad, PBSTR, rpad, "");
            fflush(stdout);
            mini_batch(training_data, training_label, params.batchsize, batch * params.batchsize);
        }

        // evaluate test data
        int correct = 0;
        double test_loss = 0.0f;
        evaluate(test_data, test_label, correct, test_loss);
        printf(" %d/%d, Train loss: %lf, Test loss: %lf\n", correct, (int)test_data.size(), loss, test_loss);
    }
}

void NeuralNetwork::mini_batch(std::vector<double *> &training_data, std::vector<double *> &training_label, int batch_size, int start)
{
    int end = start + batch_size;
    if (end > training_data.size())
    {
        std::cout << "Invalid batch size" << std::endl;
        return;
    }

    // initialize dC_dw and dC_dw
    int n = params.inputsize;
    for (int l = 0; l < params.num_layers; l++)
    {
        CUDA_CHECK(cudaMemset(dC_dw[l], 0, sizeof(double) * n * params.layers[l]));
        CUDA_CHECK(cudaMemset(dC_dw[l], 0, sizeof(double) * params.layers[l]));
        n = params.layers[l];
    }

    for (int i = start; i < end; i++)
    {
        auto output = forward(training_data[i], params.inputsize);
        for (int j = 0; j < params.outputsize; j++)
        {
            loss += 0.5f * (training_label[i][j] - output[j]) * (training_label[i][j] - output[j]);
        }
        backprop(training_label[i]);
    }

    // update weights and biases
    update_weights_and_biases();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void NeuralNetwork::evaluate(std::vector<double *> &test_data, std::vector<int> &test_label, int &correct, double& test_loss)
{
    for (int i = 0; i < test_data.size(); i++)
    {
        double *output = forward(test_data[i], params.inputsize);
        int max_index = 0;
        for (int j = 0; j < params.outputsize; j++)
        {
            if(j == test_label[i])
                test_loss += (1.0f - output[j]) * (1.0f - output[j]);
            else
                test_loss += output[j] * output[j];
            if (output[j] > output[max_index])
                max_index = j;
        }
        if (max_index == test_label[i])
            correct++;
    }
    test_loss /= 2;
}

void NeuralNetwork::save()
{
    std::ofstream file;
    // system time stamp
    std::time_t t = std::time(0);
    std::tm *now = std::localtime(&t);
    std::string w_filename = "weights_" + std::to_string(now->tm_year + 1900) + "_" + std::to_string(now->tm_mon + 1) + "_" + std::to_string(now->tm_mday) + "_" + std::to_string(now->tm_hour) + "_" + std::to_string(now->tm_min) + "_" + std::to_string(now->tm_sec) + ".csv";
    std::string b_filename = "biases_" + std::to_string(now->tm_year + 1900) + "_" + std::to_string(now->tm_mon + 1) + "_" + std::to_string(now->tm_mday) + "_" + std::to_string(now->tm_hour) + "_" + std::to_string(now->tm_min) + "_" + std::to_string(now->tm_sec) + ".csv";

    std::vector<double *> w, b;
    int n = params.inputsize;
    for (int i = 0; i < params.num_layers; i++)
    {
        w.push_back(new double[n * params.layers[i]]);
        b.push_back(new double[params.layers[i]]);
        n = params.layers[i];
    }
    _debug_get_weights_and_biases(w, b);
    file.open(w_filename);
    if (!file.is_open())
    {
        std::cout << "Cannot open file:" << w_filename << std::endl;
        return;
    }
    n = params.inputsize;
    for (int i = 0; i < params.num_layers; i++)
    {
        for (int j = 0; j < params.layers[i]; j++)
        {
            for (int k = 0; k < n; k++)
            {
                file << w[i][k * params.layers[i-1] + j] << ",";
            }
            file << std::endl;
        }
        n = params.layers[i];
    }
    file.close();
    file.open(b_filename);
    if (!file.is_open())
    {
        std::cout << "Cannot open file:" << b_filename << std::endl;
        return;
    }
    for (int i = 0; i < params.num_layers; i++)
    {
        for (int j = 0; j < params.layers[i]; j++)
        {
            file << b[i][j] << ",";
        }
        file << std::endl;
    }
    file.close();
}

#ifdef DEBUG
void NeuralNetwork::_debug_get_weights_and_biases(std::vector<double *> w, std::vector<double *> b)
{
    // copy weights and biases to host
    int n = params.inputsize;
    for (int i = 0; i < params.num_layers; i++)
    {
        CUDA_CHECK(cudaMemcpy(w[i], this->w[i], sizeof(double) * n * params.layers[i], cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b[i], this->b[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
        n = params.layers[i];
    }
}

void NeuralNetwork::_debug_get_grad(std::vector<double *> dw, std::vector<double *> db)
{
    // copy weights and biases to host
    int n = params.inputsize;
    for (int i = 0; i < params.num_layers; i++)
    {
        CUDA_CHECK(cudaMemcpy(dw[i], this->dC_dw[i], sizeof(double) * n * params.layers[i], cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(db[i], this->dC_db[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
        n = params.layers[i];
    }
}

Params NeuralNetwork::_debug_params()
{
    return params;
}

void NeuralNetwork::_debug_get_a(std::vector<double *> a)
{
    // copy a values to host
    for (int i = 0; i < params.num_layers; i++)
    {
        CUDA_CHECK(cudaMemcpy(a[i], this->a[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
    }
}

void NeuralNetwork::_debug_get_delta(std::vector<double *> delta)
{
    // copy a values to host
    for (int i = 0; i < params.num_layers; i++)
    {
        CUDA_CHECK(cudaMemcpy(delta[i], this->dC_dz[i], sizeof(double) * params.layers[i], cudaMemcpyDeviceToHost));
    }
}

void NeuralNetwork::_debug_set(std::vector<double *> w, std::vector<double *> b)
{

    if (w.size() != params.num_layers || b.size() != params.num_layers)
    {
        std::cout << "Invalid weight or bias vector size" << std::endl;
        return;
    }

    // copy weights and biases to device
    int n = params.inputsize;
    for (int i = 0; i < params.num_layers; i++)
    {
        CUDA_CHECK(cudaMemcpy(this->w[i], w[i], sizeof(double) * n * params.layers[i], cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(this->b[i], b[i], sizeof(double) * params.layers[i], cudaMemcpyHostToDevice));
        n = params.layers[i];
    }
}
#endif