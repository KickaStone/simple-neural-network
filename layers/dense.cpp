//
// Created by JunchengJi on 7/30/2023.
//
#include "dense.h"

Dense::Dense(int input_size, int output_size, Activation::ActivationFunctionType type) : Layer(input_size, output_size, type) {
    w = Eigen::MatrixXd::Zero(output_size, input_size);
    b = Eigen::VectorXd::Zero(output_size);
    a = Eigen::VectorXd::Zero(output_size);
    dw = Eigen::MatrixXd::Zero(output_size, input_size);
    db = Eigen::VectorXd::Zero(output_size);
    dz = Eigen::VectorXd::Zero(output_size);
    input_grad = Eigen::VectorXd::Zero(input_size);

    // initialize w and b
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    w = w.unaryExpr([&](double dummy){return d(gen);});
}

Dense::~Dense() {
}

double* Dense::forward(const double *input_data) {
    input = input_data;
    Eigen::Map<const Vec> x = Eigen::Map<const Vec>(input_data, inputSize);
    a.noalias() = w * x + b;
    a.noalias() = a.unaryExpr(std::ref(activation));
    return a.data();
}

double* Dense::backward(const double *output_grad) {
    Eigen::Map<const Vec> grad(output_grad, outputSize);
    Eigen::Map<const Vec> x(input, inputSize);
    dz = grad.cwiseProduct(a.unaryExpr(std::ref(derivative)));
    input_grad.noalias() = w.transpose() * dz;
    dw += dz * x.transpose();
    db += dz;
    return input_grad.data();
}

void Dense::update(double lr, int batchSize) {
    double scaler = lr / batchSize;
    w -= dw * scaler;
    b -= db * scaler;
    dw.setZero();
    db.setZero();
}