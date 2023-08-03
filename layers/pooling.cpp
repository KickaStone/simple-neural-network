//
// Created by JunchengJi on 7/31/2023.
//

#include "pooling.h"

Pooling::Pooling(int inputChannel, int inputHeight, int inputWidth, int kenrelSize, int stride, int padding, PoolingType type)
    : Layer(inputChannel * inputHeight * inputWidth,
            inputChannel * ((inputHeight + 2 * padding - kenrelSize) / stride + 1) * ((inputWidth + 2 * padding - kenrelSize) / stride + 1),
            Activation::ActivationFunctionType::NONE),
      _inputChannel(inputChannel),
      _inputHeight(inputHeight),
      _inputWidth(inputWidth),
      _kenrelSize(kenrelSize),
      _stride(stride),
      _padding(padding),
      _outputHeight((inputHeight + 2 * padding - kenrelSize) / stride + 1),
      _outputWidth((inputWidth + 2 * padding - kenrelSize) / stride + 1),
      poolingType(type)
{
    _output_v = new double[_outputHeight * _outputWidth * _inputChannel];
    _output_m = std::vector<Mat>(_inputChannel, Mat::Zero(_outputHeight, _outputWidth));
    if(PoolingType::MAX == poolingType){
        _record = std::vector<Mat>(_inputChannel, Mat::Zero(_outputHeight, _outputWidth));
    }
    _input_m = std::vector<MatMap>(_inputChannel, MatMap(nullptr, _inputHeight, _inputWidth));
    _input_grad_v = new double[_inputHeight * _inputWidth * _inputChannel];
    _input_grad_m = std::vector<Mat>(_inputChannel, Mat::Zero(_inputHeight, _inputWidth));
}

Pooling::~Pooling()
{
    delete[] _output_v;
}

void max_pooling(const MatMap &input, Mat &output, Mat &record, int kernelSize, int stride, int padding)
{
    int inputHeight = input.rows();
    int inputWidth = input.cols();
    int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;
    
    for (int i = 0; i < outputHeight; i++)
    {
        for (int j = 0; j < outputWidth; j++)
        {
            int startRow = i * stride - padding;
            int startCol = j * stride - padding;
            int endRow = std::min(startRow + kernelSize, inputHeight);
            int endCol = std::min(startCol + kernelSize, inputWidth);
            startRow = std::max(startRow, 0);
            startCol = std::max(startCol, 0);

            if(startRow >= endRow || startCol >= endCol){
                output(i, j) = 0;
                continue;
            }else{
                int max_i = 0, max_j = 0;
                output(i, j) = input.block(startRow, startCol, endRow - startRow, endCol - startCol).maxCoeff(&max_i, &max_j);
                record(i, j) = (startRow + max_i) * inputWidth + startCol + max_j;
            }
        }
    }
}

void max_pooling_backprop(const Mat &dz, const Mat &record, Mat &grad){
    for(int i = 0; i < dz.rows(); i++){
        for(int j = 0; j < dz.cols(); j++){
            int index = record(i, j);
            grad(index / grad.cols(), index % grad.cols()) = dz(i, j);
        }
    }
}

void avg_pooling(const MatMap &input, Mat &output, int kernelSize, int stride, int padding)
{
    int inputHeight = input.rows();
    int inputWidth = input.cols();
    int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

    for (int i = 0; i < outputHeight; i++)
    {
        for (int j = 0; j < outputWidth; j++)
        {
            int startRow = i * stride - padding;
            int startCol = j * stride - padding;
            int endRow = std::min(startRow + kernelSize, inputHeight);
            int endCol = std::min(startCol + kernelSize, inputWidth);
            startRow = std::max(startRow, 0);
            startCol = std::max(startCol, 0);

            if(startRow >= endRow || startCol >= endCol){
                output(i, j) = 0;
                continue;
            }else{
                output(i, j) = input.block(startRow, startCol, endRow - startRow, endCol - startCol).sum() / (kernelSize * kernelSize);
            }
        }
    }
}

void avg_pooling_backprop(const Mat &dz, Mat &grad, int kernelSize, int stride, int padding){
    int inputHeight = grad.rows();
    int inputWidth = grad.cols();
    int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

    for(int i = 0; i < outputHeight; i++){
        for(int j = 0; j < outputWidth; j++){
            int startRow = i * stride - padding;
            int startCol = j * stride - padding;
            int endRow = std::min(startRow + kernelSize, inputHeight);
            int endCol = std::min(startCol + kernelSize, inputWidth);
            startRow = std::max(startRow, 0);
            startCol = std::max(startCol, 0);

            if(startRow >= endRow || startCol >= endCol){
                continue;
            }else{
                grad.block(startRow, startCol, endRow - startRow, endCol - startCol).array() += dz(i, j) / (kernelSize * kernelSize);
            }
        }
    }
}

double *Pooling::forward(const double *input){
    for(int i = 0; i < _inputChannel; i++){
        new (&_input_m[i]) MatMap(input + i * _inputHeight * _inputWidth, _inputHeight, _inputWidth);
    }

    switch (poolingType)
    {
        case PoolingType::MAX:
            for(int i = 0; i < _inputChannel; i++){
                max_pooling(_input_m[i], _output_m[i], _record[i], _kenrelSize, _stride, _padding);
                // std::cout << _record[i] << std::endl;
            }
            break;
        case PoolingType::AVG:
            for(int i = 0; i < _inputChannel; i++){
                avg_pooling(_input_m[i], _output_m[i], _kenrelSize, _stride, _padding);
            }
            break;

        default:
            break;
    }

    for(int i = 0; i < _inputChannel; i++){
        std::copy(_output_m[i].data(), _output_m[i].data() + _outputHeight * _outputWidth, _output_v + i * _outputHeight * _outputWidth);
    }
    return _output_v;
}

double *Pooling::backward(const double *grad)
{
    std::vector<MatMap> output_gard;
    for(int i = 0; i < _inputChannel; i++){
        output_gard.push_back(MatMap(grad + i * _outputHeight * _outputWidth, _outputHeight, _outputWidth));
    }
    // std::cout << "output_grad: " << std::endl;
    // for(int i = 0; i < _inputChannel; i++){
    //     std::cout << output_gard[i] << std::endl;
    // } 
    // reshape
    // backprop
    switch (poolingType)
    {
        case PoolingType::MAX:
            for(int i = 0; i < _inputChannel; i++){
                max_pooling_backprop(output_gard[i], _record[i], _input_grad_m[i]);
            }
            break;
        case PoolingType::AVG:
            for(int i = 0; i < _inputChannel; i++){
                avg_pooling_backprop(output_gard[i], _input_grad_m[i], _kenrelSize, _stride, _padding);
            }
            break;
        default:
            break;
    }

    // copy to _input_grad_v
    for(int i = 0; i < _inputChannel; i++){
        std::copy(_input_grad_m[i].data(), _input_grad_m[i].data() + _inputHeight * _inputWidth, _input_grad_v + i * _inputHeight * _inputWidth);
    }
    return _input_grad_v;
}

void Pooling::update(double lr, int batchSize)
{
    // do nothing
}
