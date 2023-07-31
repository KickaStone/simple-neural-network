#include "conv.h"

Conv::Conv(int Nx, int Ny, int kernel_size, int stride, int n_kernel, int padding, Activation::ActivationFunctionType type)
           : Layer(Nx*Ny,
                   n_kernel * ((Nx + 2*padding - kernel_size) / stride + 1) * ((Ny + 2*padding - kernel_size) / stride + 1),
                   type) {
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
    a = new double[Ox * Oy * n_kernel];
    dk = std::vector<double*>(n_kernel);
    db = new double[n_kernel];
    dz = new double[outputSize];

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
    delete[] a;
    for(auto fm : feature_map){
        delete[] fm;
    }
    for(auto d : dk){
        delete[] d;
    }
    delete[] dz;
    delete[] db;
}

double *Conv::forward(const double *data) {
    this->input = data;
    for(int i = 0; i < n_kernel; i++){
        convolution::cross_correlation(data, kernel[i], Nx, Ny, kernel_size, stride, feature_map[i]);
    }
    // calculate a
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Ox * Oy; j++){
            a[i * Ox * Oy + j] = activation(feature_map[i][j] + b[i]);
        }
    }
    return a;
}

double *Conv::backward(const double *grad) {
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Ox * Oy; j++) {
            dz[i * Ox * Oy + j] = grad[i * Ox * Oy + j] * derivative(a[i * Ox * Oy + j]);
        }
    }

    auto *tmp = new double [Nx * Ny];

    // calculate gradient of kernel
    // DK = dz (*) input
    for(int i = 0; i < n_kernel; i++){
        auto* dki = new double[kernel_size * kernel_size];
        convolution::cross_correlation(input, dz + i * Ox * Oy, Nx, Ny, Ox, stride, dki);
        for(int j = 0; j < kernel_size*kernel_size; j++){
            dk[i][j] += dki[j];
        }
        delete[] dki;
    }

    // calculate gradient of bias
    // db = sum(dz)
    for(int i = 0; i < n_kernel; i++){
        for(int j = 0; j < Ox * Oy; j++){
            db[i] += dz[i * Ox * Oy + j];
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
        convolution::padding(dz + i * Ox * Oy,Ox, P, dz_padded + i * Ndz * Ndz);
        convolution::correlation(dz_padded + i * Ndz * Ndz, kernel[i], Ndz, Ndz, kernel_size, 1, input_grad + i * Nx * Ny);
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

void Conv::setKernel(int i, double *myKernel) {
    std::copy(myKernel, myKernel + kernel_size * kernel_size, this->kernel[i]);
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

void Conv::setInput(double *x) {
    this->input = x;
}


