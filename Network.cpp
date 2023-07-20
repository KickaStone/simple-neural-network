#include "Network.h"

double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }
double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

double relu(double z) { return z > 0 ? z : 0; }
double relu_prime(double z) { return z > 0 ? 1 : 0; }

Network::Network(int *sizes, int num_layers)
{
    this->num_layers = num_layers;
    this->sizes = (int *)malloc(sizeof(int) * num_layers);
    int num_bias = 0;
    for (int i = 0; i < num_layers; i++)
    {
        this->sizes[i] = sizes[i];
        if (i > 0)
            num_bias += sizes[i];
    }

    this->bias = (double **)malloc(sizeof(double) * (num_layers - 1));
    this->weight = (double **)malloc(sizeof(double *) * (num_layers - 1));
    for (int i = 0; i < num_layers - 1; i++)
    {
        this->bias[i] = (double *)malloc(sizeof(double) * sizes[i + 1]);
        this->weight[i] = (double *)malloc(sizeof(double) * sizes[i + 1] * sizes[i]);
    }

    init();
}

Network::Network(){
    this->num_layers = 0;
    this->sizes = NULL;
    this->bias = NULL;
    this->weight = NULL;
}

Network::~Network()
{
    free(this->sizes);
    free(this->bias);
    for(int i = 0; i < num_layers - 1; i++){
        free(this->weight[i]);
    }
    free(this->weight);
}

void Network::print()
{
    // print size
    printf("num_layers: %d\n", num_layers);
    printf("sizes: ");
    for (int i = 0; i < num_layers; i++)
    {
        printf("%d ", sizes[i]);
    }
    printf("\n");
}

bool Network::checkBiasIdx(int layer, int index)
{
    return layer > 0 && layer < num_layers && index >= 0 && index < sizes[layer];
}

bool Network::checkWeightIdx(int layer, int j, int k)
{
    return layer > 0 && layer < num_layers && j >= 0 && j < sizes[layer] && k >= 0 && k < sizes[layer - 1];
}

void Network::init()
{
    // random init bias and weight
    // standard normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    for(int i = 0; i < num_layers - 1; i++){
        for(int j = 0; j < sizes[i+1]; j++){
            bias[i][j] = d(gen);
            for(int k = 0; k < sizes[i]; k++){
                weight[i][j * sizes[i] + k] = d(gen);
            }
        }
    }
}

void Network::saveNetwork()
{
    // system time as file name 
    time_t t = time(NULL);
    char filename[32];
    strftime(filename, sizeof(filename), "%Y%m%d%H%M%S", localtime(&t));
    FILE *fp = fopen(filename, "wb");
    if(fp == NULL){
        printf("Error: cannot open file %s\n", filename);
        return;
    }

    // write num_layers
    fwrite(&num_layers, sizeof(int), 1, fp);

    // write sizes
    fwrite(sizes, sizeof(int), num_layers, fp);

    // write bias  
    for(int i = 0; i < num_layers - 1; i++){
        fwrite(bias[i], sizeof(double), sizes[i+1], fp);
    }

    // write weight
    for(int i = 0; i < num_layers - 1; i++){
        fwrite(weight[i], sizeof(double), sizes[i+1] * sizes[i], fp);
    }

    fclose(fp);
}

void Network::loadNetwork()
{
    // select file
    char filename[32];
    printf("Please input the file name: ");
    scanf("%s", filename);
    FILE *fp = fopen(filename, "rb");
    if(fp == NULL){
        printf("Error: cannot open file %s\n", filename);
        return;
    }

    // read num_layers
    fread(&num_layers, sizeof(int), 1, fp);

    // read sizes
    sizes = (int *)malloc(sizeof(int) * num_layers);
    fread(sizes, sizeof(int), num_layers, fp);

    // read bias
    bias = (double **)malloc(sizeof(double *) * (num_layers - 1));
    for(int i = 0; i < num_layers - 1; i++){
        bias[i] = (double *)malloc(sizeof(double) * sizes[i+1]);
        fread(bias[i], sizeof(double), sizes[i+1], fp);
    }

    // read weight
    weight = (double **)malloc(sizeof(double *) * (num_layers - 1));
    for(int i = 0; i < num_layers - 1; i++){
        weight[i] = (double *)malloc(sizeof(double) * sizes[i+1] * sizes[i]);
        fread(weight[i], sizeof(double), sizes[i+1] * sizes[i], fp);
    }

    fclose(fp);
}

double Network::getBias(int layer, int index)
{
    if(checkBiasIdx(layer, index) == false){
        printf("Try to get {%d, %d} bias, out of range\n", layer, index);
        return 0.0;
    }
    return bias[layer-1][index];
}

/**
 * @brief Get the Weight object
 * @param layer the output node layer
 * @param j the output node index
 * @param k the input node index
 */
double Network::getWeight(int layer, int j, int k)
{
    if(checkWeightIdx(layer, j, k) == false){
        printf("Try to get {%d, %d, %d} weigth, out of range\n", layer, j, k);
        return 0.0;
    }
    return weight[layer - 1][j * sizes[layer - 1] + k];
}

void Network::setBias(int layer, int index, double value)
{
    if(checkBiasIdx(layer, index) == false){
        printf("Try to set {%d, %d} bias, out of range\n", layer, index);
        return;
    }
    bias[layer-1][index] = value;
}

void Network::setWeight(int layer, int j, int k, double value)
{
    if (checkWeightIdx(layer, j, k) == false)
    {
        printf("Try to set {%d, %d, %d} weigth, out of range\n", layer, j, k);
        return;
    }
    weight[layer - 1][j * sizes[layer - 1] + k] = value;
}

/**
 * @brief feedforward
 * @param a the activation of input layer
 * @return the output layer
 */
double* Network::feedforward(const double* a){
    double *in_a = (double*)malloc(sizeof(double) * sizes[0]);
    // for loop copy
    for(int i = 0; i < sizes[0]; i++){
        in_a[i] = a[i];
    }

    for(int i = 0; i < num_layers - 1; i++){
        double *out_a = (double*)malloc(sizeof(double) * sizes[i+1]);
        for(int j = 0; j < sizes[i+1]; j++){
            double z = 0.0;
            for(int k = 0; k < sizes[i]; k++){
                z += getWeight(i+1, j, k) * in_a[k];
            }
            z += getBias(i+1, j);
            out_a[j] = sigmoid(z);
        }
        free(in_a);
        in_a = out_a;
    }
    
    return in_a;
}

/**
 * @brief backprop
 * @param input the input layer
 * @param output the output layer
 * @return the nabla of bias and weight
*/
nabla Network::backprop(const double *input, unsigned int label)
{
    double* output = (double*)malloc(sizeof(double) * sizes[num_layers-1]);
    for(int i = 0; i < sizes[num_layers-1]; i++){
        output[i] = 0.0;
    }
    output[label] = 1.0;

    nabla nb;
    nb.nabla_bias = (double **)malloc(sizeof(double *) * (num_layers - 1));
    nb.nabla_weight = (double **)malloc(sizeof(double *) * (num_layers - 1));
    for (int i = 0; i < num_layers - 1; i++)
    {
        nb.nabla_bias[i] = (double *)malloc(sizeof(double) * sizes[i + 1]);
        nb.nabla_weight[i] = (double *)malloc(sizeof(double) * sizes[i + 1] * sizes[i]);
    }

    // feedforward
    double **activation = (double **)malloc(sizeof(double *) * num_layers); // activation
    double **z = (double **)malloc(sizeof(double *) * num_layers); // store all z vectors, layer by layer
    activation[0] = (double *)malloc(sizeof(double) * sizes[0]); 
    memcpy(activation[0], input, sizeof(double) * sizes[0]);


    for (int i = 0; i < num_layers - 1; i++)
    {
        activation[i + 1] = (double *)malloc(sizeof(double) * sizes[i + 1]);
        z[i + 1] = (double *)malloc(sizeof(double) * sizes[i + 1]);
        for (int j = 0; j < sizes[i + 1]; j++)
        {
            double z_value = 0.0;
            for (int k = 0; k < sizes[i]; k++)
            {
                z_value += getWeight(i + 1, j, k) * activation[i][k];
            }
            z_value += getBias(i + 1, j);
            z[i + 1][j] = z_value;
            activation[i + 1][j] = sigmoid(z_value);
        }
    }

    // calculate loss
    for (int i = 0; i < sizes[num_layers - 1]; i++)
    {
        loss += 0.5 * pow(activation[num_layers - 1][i] - output[i], 2);
    }
    // backward pass
    // loss = 1/2 * (a[L] - y)^2
    // delta = (a[L] - y) * sigmoid_prime(z[L])
    double *delta = (double *)malloc(sizeof(double) * sizes[num_layers - 1]);
    for (int i = 0; i < sizes[num_layers - 1]; i++)
    {
        delta[i] = cost_derivative(activation[num_layers - 1][i], output[i]) * sigmoid_prime(z[num_layers - 1][i]);
    }

    // nabla_b = delta
    for (int i = 0; i < sizes[num_layers - 1]; i++)
    {
        nb.nabla_bias[num_layers - 2][i] = delta[i];
    }

    // nabla_w[l] = delta[l] * a[l-1]
    for (int i = 0; i < sizes[num_layers - 1]; i++)
    {
        for (int j = 0; j < sizes[num_layers - 2]; j++)
        {
            nb.nabla_weight[num_layers - 2][i * sizes[num_layers - 2] + j] = delta[i] * activation[num_layers - 2][j];
        }
    }

    // L = num_layers - 2
    for(int L = num_layers - 2; L > 0; L--){
        // delta = ((w[L+1])^T * delta[L+1]) * sigmoid_prime(z[L])
        double *delta_L = (double *)malloc(sizeof(double) * sizes[L]);
        for(int i = 0; i < sizes[L]; i++){
            double delta_L_value = 0.0;
            for(int j = 0; j < sizes[L+1]; j++){
                delta_L_value += getWeight(L+1, j, i) * delta[j];
            }
            delta_L[i] = delta_L_value * sigmoid_prime(z[L][i]);
        }
        free(delta);
        delta = delta_L;

        // nabla_b = delta
        for(int i = 0; i < sizes[L]; i++){
            nb.nabla_bias[L-1][i] = delta[i];
        }

        // nabla_w = delta * a[L-1]
        for(int i = 0; i < sizes[L]; i++){
            for(int j = 0; j < sizes[L-1]; j++){
                nb.nabla_weight[L-1][i * sizes[L-1] + j] = delta[i] * activation[L-1][j];
            }
        }
    }

    free(delta);
    for(int i = 0; i < num_layers; i++){
        free(activation[i]);
        if(i) free(z[i]);
    }
    free(activation);
    free(z);
    return nb;
}

double Network::cost_derivative(double output, double y)
{
    return output - y;
}

int Network::evaluate(double **data, unsigned int *label, int n){
    int correct = 0;
    for(int i = 0; i < n; i++){
        double *output = feedforward(data[i]);
        int max_idx = 0;
        for(int j = 0; j < sizes[num_layers-1]; j++){
            if(output[j] > output[max_idx]){
                max_idx = j;
            }
        }
        if(max_idx == label[i]){
            correct++;
        }
        free(output);
    }
    return correct;
}

/**
 * @brief SGD
 * @param data the training data
 * @param label the training label
 * @param epochs the number of epochs
 * @param mini_batch_size the size of mini batch
 * @param eta the learning rate
 * @param test_data the test data
 * @param test_label the test label
 * @param num_training_data the number of training data
 * @param num_test_data the number of test data
 * @return void
*/
void Network::SGD(double **data, unsigned int *label, int epochs, int mini_batch_size, double eta, double **test_data, unsigned int *test_label, int num_training_data, int num_test_data)
{
    for(int i = 0; i < epochs; i++){
        // shuffle the training data
        srand(time(NULL));
        for(int j = 0; j < num_training_data; j++){
            int idx = rand() % num_training_data;
            double *tmp = data[j];
            data[j] = data[idx];
            data[idx] = tmp;
            unsigned int tmp_label = label[j];
            label[j] = label[idx];
            label[idx] = tmp_label;
        }

        // update mini batch
        for(int j = 0; j < num_training_data; j += mini_batch_size){
            update_mini_batch(data, label, j, mini_batch_size, eta);
        }
        printf("loss: %lf\n", loss);
        loss = 0.0;
        printf("Epoch %d: %d / %d\n", i, evaluate(test_data, test_label, num_test_data), num_test_data);
    }
}

void Network::update_mini_batch(double **data, unsigned int *label, int start, int mini_batch_size, double eta)
{
    nabla nb;
    nb.nabla_bias = (double **)malloc(sizeof(double *) * (num_layers - 1));
    nb.nabla_weight = (double **)malloc(sizeof(double *) * (num_layers - 1));
    for (int i = 0; i < num_layers - 1; i++)
    {
        nb.nabla_bias[i] = (double *)malloc(sizeof(double) * sizes[i + 1]);
        nb.nabla_weight[i] = (double *)malloc(sizeof(double) * sizes[i + 1] * sizes[i]);
    }

    // initiliza nabla
    for(int i = 0; i < num_layers - 1; i++){
        for(int j = 0; j < sizes[i+1]; j++){
            nb.nabla_bias[i][j] = 0.0;
            for(int k = 0; k < sizes[i]; k++){
                nb.nabla_weight[i][j * sizes[i] + k] = 0.0;
            }
        }
    }

    for(int i = 0; i < mini_batch_size; i++){
        nabla nb_i = backprop(data[i + start], label[i + start]);
        for(int j = 0; j < num_layers - 1; j++){
            for(int k = 0; k < sizes[j+1]; k++){
                nb.nabla_bias[j][k] += nb_i.nabla_bias[j][k];
                for(int l = 0; l < sizes[j]; l++){
                    nb.nabla_weight[j][k * sizes[j] + l] += nb_i.nabla_weight[j][k * sizes[j] + l];
                }
            }
        }
    }

    for(int i = 0; i < num_layers - 1; i++){
        for(int j = 0; j < sizes[i+1]; j++){
            setBias(i+1, j, getBias(i+1, j) - eta / mini_batch_size * nb.nabla_bias[i][j]);
            for(int k = 0; k < sizes[i]; k++){
                setWeight(i+1, j, k, getWeight(i+1, j, k) - eta / mini_batch_size * nb.nabla_weight[i][j * sizes[i] + k]);
            }
        }
    }

    for(int i = 0; i < num_layers - 1; i++){
        free(nb.nabla_bias[i]);
        free(nb.nabla_weight[i]);
    }
    free(nb.nabla_bias);
    free(nb.nabla_weight);
}
