//
// Created by JunchengJi on 7/31/2023.
//

#include "pooling.h"

pooling::pooling(int inputChannel, int inputHeight, int inputWidth, int kenrelSize, int stride, int padding, PoolingType type)
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
    if(PoolingType::MAX_POOL_2D == poolingType){
        _record = std::vector<Mat>(_inputChannel, Mat::Zero(_outputHeight, _outputWidth));
    }
}

pooling::~pooling()
{
    delete[] _output_v;
}


void max_pooling(const Mat &input, Mat &output, Mat &record, int kernelSize, int stride, int padding)
{
    int inputHeight = input.rows();
    int inputWidth = input.cols();
    int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

    for (int i = 0; i < outputHeight; i++)
    {
        for (int j = 0; j < outputWidth; j++)
        {
            int startRow = std::max(i * stride - padding, 0);
            int startCol = std::max(j * stride - padding, 0);
            int endRow = std::min(startRow + kernelSize, inputHeight);
            int endCol = std::min(startCol + kernelSize, inputWidth);

            double maxVal = -std::numeric_limits<double>::infinity();
            if(startRow >= endRow || startCol >= endCol){
                output(i, j) = 0;
                continue;
            }else{
                int max_i = 0, max_j = 0;
                output(i, j) = input.block(startRow, startCol, endRow - startRow, endCol - startCol).maxCoeff(&max_i, &max_j);
                record(i, j) = max_i * (endCol - startCol) + max_j;
            }
        }
    }
}

void avg_pooling(const Mat &input, Mat &output, int kernelSize, int stride, int padding)
{
    int inputHeight = input.rows();
    int inputWidth = input.cols();
    int outputHeight = (inputHeight + 2 * padding - kernelSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernelSize) / stride + 1;

    for (int i = 0; i < outputHeight; i++)
    {
        for (int j = 0; j < outputWidth; j++)
        {
            int startRow = std::max(i * stride - padding, 0);
            int startCol = std::max(j * stride - padding, 0);
            int endRow = std::min(startRow + kernelSize, inputHeight);
            int endCol = std::min(startCol + kernelSize, inputWidth);

            if(startRow >= endRow || startCol >= endCol){
                output(i, j) = 0;
                continue;
            }else{
                output(i, j) = input.block(startRow, startCol, endRow - startRow, endCol - startCol).sum() / (kernelSize * kernelSize);
            }
        }
    }
}

double *pooling::forward(const double *input){
    for(int i = 0; i < _inputChannel; i++){
        reshape_matrix(input + i * _inputHeight * _inputWidth, _input_m[i], _inputHeight, _inputWidth);
    }

    switch (poolingType)
    {
        case PoolingType::MAX_POOL_2D:
            for(int i = 0; i < _inputChannel; i++){
                max_pooling(_input_m[i], _output_m[i], _record[i], _kenrelSize, _stride, _padding);
            }
            break;
        case PoolingType::AVG_POOL_2D:
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

double *pooling::backward(const double *grad)
{
    
    // TODO: implement backward
    return nullptr;
}

void pooling::update(double lr, int batchSize)
{
    // do nothing
}
