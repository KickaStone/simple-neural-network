#include "conv.h"

#include <utility>

void Conv::cross_correlation(double *img, double *kernel, int Nx, int Nk, int stride, double *output){
    int Ox = (Nx - Nk) / stride + 1;
    for(int i = 0; i < Ox; i++){
        for(int j = 0; j < Ox; j++){
            double sum = 0;
            for(int k = 0; k < Nk; k++){
                for(int l = 0; l < Nk; l++){
                    sum += img[(i + k) * Nx + (j + l)] * kernel[k * Nk + l];
                }
            }
            output[i * Ox + j] = sum;
        }
    }
}

void Conv::correlation(double *img, double *kernel, int Nx, int Nk, int stride, double *output){
    int Ox = (Nx - Nk) / stride + 1;
    for(int i = 0; i < Ox; i++){
        for(int j = 0; j < Ox; j++){
            double sum = 0;
            for(int k = 0; k < Nk; k++){
                for(int l = 0; l < Nk; l++){
                    sum += img[(i + k) * Nx + (j + l)] * kernel[(Nk - k - 1) * Nk + (Nk - l - 1)];
                }
            }
            output[i * Ox + j] = sum;
        }
    }
}

void Conv::padding(double *input, int Nx, int p, double *output){
    int Nx_p = Nx + 2 * p;
    for(int i = 0; i < Nx_p; i++){
        for(int j = 0; j < Nx_p; j++){
            if(i < p || i >= Nx_p - p || j < p || j >= Nx_p - p){
                output[i * Nx_p + j] = 0;
            } else {
                output[i * Nx_p + j] = input[(i - p) * Nx + (j - p)];
            }
        }
    }
}

Conv::Conv(int inputSize, Activation::ActivationFunctionType type, int Nx, int Ny, int kernel_size,
           int stride, int n_kernel, int padding) : Layer(inputSize, (inputSize + 2*padding - kernel_size) / stride + 1, type) {
    this->n_kernel = n_kernel;
    this->input = nullptr;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->p = padding;
    this->Nx = Nx;
    this->Ny = Ny;
    this->Ox = (Nx + 2*padding - kernel_size) / stride + 1;
    this->Oy = (Ny + 2*padding - kernel_size) / stride + 1;

    kernel = std::vector<double*>(n_kernel);
    feature_map = std::vector<double*>(n_kernel);
    for (int i = 0; i < n_kernel; ++i) {
        feature_map[i] = new double[Ox * Oy];
        kernel[i] = new double[kernel_size * kernel_size];
    }

    b = new double[n_kernel];
    output = new double[Ox * Oy * n_kernel];
    dk = std::vector<double*>(n_kernel);
    db = new double[n_kernel];
    dz = new double[Nx * Ny * n_kernel];

    // Initialize kernel
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    for (int i = 0; i < n_kernel; i++) {
        auto* k = new double[kernel_size * kernel_size];
        for (int j = 0; j < kernel_size * kernel_size; j++) {
            k[j] = dis(gen);
        }
    }

    // Initialize bias
    std::fill(b, b + n_kernel, 0);

    // Initialize dk
    for(int i = 0; i < n_kernel; i++){
        dk[i] = new double[kernel_size * kernel_size];
        std::fill(dk[i], dk[i] + kernel_size * kernel_size, 0);
        db[i] = 0;
    }
}

Conv::~Conv() {
    for(double* k : kernel){
        delete[] k;
    }
    delete[] b;
    delete[] output;
    for(auto fm : feature_map){
        delete[] fm;
    }
    for(auto d : dk){
        delete[] d;
    }
    delete[] dz;
    delete[] db;
}

double *Conv::forward(double *input) {
    this->input = input;
    for(int i = 0; i < n_kernel; i++){
        cross_correlation(input, kernel[i], Nx, kernel_size, stride, feature_map[i]);
    }
    // calculate output
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Ox * Oy; j++){
            output[i * Ox * Oy + j] = activation(feature_map[i][j] + b[i]);
        }
    }
    return output;
}

double *Conv::backward(double *grad) {
    double *tmp = new double [Nx * Ny];
    // calculate gradient of kernel
    // dz = grad * activationFunc.derivative(output)
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Ox * Oy; j++) {
            grad[i * Ox * Oy + j] *= activationDerivative(output[i * Ox * Oy + j]);
        }
    }

    // calculate gradient of kernel
    // DK = dz (*) input
    for(int i = 0; i < n_kernel; i++){
        auto* dki = new double[kernel_size * kernel_size];
        cross_correlation(input, grad + i * Ox * Oy, Nx, Ox, stride, dki);
        for(int j = 0; j < kernel_size*kernel_size; j++){
            dk[i][j] += dki[j];
        }
        delete[] dki;
    }

    // calculate gradient of bias
    // db = sum(dz)
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Ox * Oy; j++){
            db[i] += grad[i * Ox * Oy + j];
        }
    }

    // calculate gradient of input
    // dz = dz (*) kernel, full correlation
    int P = kernel_size - 1; // padding size
    int Ndz = Ox + 2 * P; // size of padded dz
    auto* dz_padded = new double[Ndz * Ndz * n_kernel];
    auto* input_grad = new double[Nx * Ny * n_kernel];
    std::fill(input_grad, input_grad + n_kernel * Nx * Ny, 0); // initialize input_grad

    for(int i = 0; i < n_kernel; i++){
        padding(grad + i * Ox * Oy,Ox, P, dz_padded + i * Ndz * Ndz);
        correlation(dz_padded + i * Ndz * Ndz, kernel[i], Ndz, kernel_size, 1, input_grad + i * Nx * Ny);
    }

    // add up
    std::fill(tmp, tmp + Nx * Ny, 0);
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Nx * Ny; j++){
            tmp[j] += input_grad[i * Nx * Ny + j];
        }
    }
    delete[] dz_padded;
    delete[] input_grad;
    return tmp;
}

void Conv::setKernel(int i, double *kernel) {
    std::copy(kernel, kernel + kernel_size * kernel_size, this->kernel[i]);
}

void Conv::setBias(int i, double bias) {
    b[i] = bias;
}

std::vector<double *> &Conv::getDk() {
    return dk;
}

void Conv::update(double lr, int batchSize) {
    double s = lr / batchSize;
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < kernel_size * kernel_size; j++){
            kernel[i][j] -= s * dk[i][j];
        }
        b[i] -= s * db[i];
    }

    // reset
    for(int i = 0; i < n_kernel; i++){
        std::fill(dk[i], dk[i] + kernel_size * kernel_size, 0);
    }
    std::fill(db, db + n_kernel, 0);
}

void Conv::setInput(double *input) {
    this->input = input;
}


