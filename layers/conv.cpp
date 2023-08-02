#include "conv.h"

Conv::Conv(int inputChannel, int inputHeight, int inputWidth, int outputChannel, int kernel_size, int stride, int padding, Activation::ActivationFunctionType type)
: Layer(inputChannel * inputHeight * inputWidth,
        outputChannel * ((inputHeight + 2*padding - kernel_size) / stride + 1) * ((inputWidth + 2*padding - kernel_size) / stride + 1),
        type),
        _inputChannel(inputChannel),
        _inputHeight(inputHeight),
        _inputWidth(inputWidth),
        _outputChannel(outputChannel),
        _kernel_size(kernel_size),
        _stride(stride),
        _padding(padding),
        _outputHeight((inputHeight + 2*padding - kernel_size) / stride + 1),
        _outputWidth((inputWidth + 2*padding - kernel_size) / stride + 1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 1);

    _output = new double[_outputHeight * _outputWidth * _outputChannel];
    _grad = new double[_inputHeight * _inputWidth * _inputChannel];

    // initialize kernel
    a = Mat3d(_outputChannel, Mat::Zero(_outputHeight, _outputWidth));
    b = Vec::Zero(outputChannel);
    K = std::vector<Mat3d>(_outputChannel);
    for(int i = 0; i < outputChannel; i++){
        K[i].resize(inputChannel, Eigen::MatrixXd::Zero(kernel_size, kernel_size));
        for(auto k : K[i])
            k = k.unaryExpr([&](double x){return dis(gen);});
    }
    db = Vec::Zero(outputChannel);
    dK = std::vector<Mat3d>(_outputChannel, Mat3d(_inputChannel, Mat::Zero(kernel_size, kernel_size)));
}

Conv::~Conv() {
    delete[] _output;
    delete[] _grad;
};

void cross_correlation(const Eigen::Ref<const Mat> &data, Mat &kernel, Mat &output, int stride, int padding){
    // std::cout << "--------------------------------" << std::endl;
    // std::cout << "data: " << std::endl << data << std::endl;
    // std::cout << "kernel: " << std::endl << kernel << std::endl;
    int inputHeight = data.rows();
    int inputWidth = data.cols();
    int outputHeight = (inputHeight + 2 * padding - kernel.rows()) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernel.cols()) / stride + 1;
    Mat x;

    if(padding > 0){
        x = Mat::Zero(inputHeight + 2 * padding, outputHeight + 2 * padding);
        x.block(padding, padding, inputHeight, inputWidth) = data;
    }else{
        x = data;
    }
    
    for(int i = 0; i < outputHeight; i++){
        for(int j = 0; j < outputWidth; j++){
            if(i * stride + kernel.rows() > x.rows() || j * stride + kernel.cols() > x.cols())
                throw "Convolution out of bound";
            output(i, j) = (x.block(i * stride, j * stride, kernel.rows(), kernel.cols()).array() * kernel.array()).sum();
        }
    }
    // std::cout << "output: " << std::endl << output << std::endl;
}

void correlation(const Eigen::Ref<const Mat> &data, const Eigen::Ref<const Mat> &kernel, Mat &output, int stride, int padding){
    // std::cout << "--------------------------------" << std::endl;
    // std::cout << "data: " << std::endl << data << std::endl;
    // std::cout << "kernel: " << std::endl << kernel << std::endl;
    int inputHeight = data.rows();
    int inputWidth = data.cols();
    int outputHeight = (inputHeight + 2 * padding - kernel.rows()) / stride + 1;
    int outputWidth = (inputWidth + 2 * padding - kernel.cols()) / stride + 1;

    Mat x;
    if(padding > 0){
        x = Mat::Zero(inputHeight + 2 * padding, outputHeight + 2 * padding);
        x.block(padding, padding, inputHeight, inputWidth) = data;
    }else{
        x = data;
    }

    for(int i = 0; i < outputHeight; i++){
        for(int j = 0; j < outputWidth; j++){
            if(i * stride + kernel.rows() > x.rows() || j * stride + kernel.cols() > x.cols()) continue;
            output(i, j) = (x.block(i * stride, j * stride, kernel.rows(), kernel.cols()).array()
                            * kernel.reverse().array()).sum();
        }
    }
    // std::cout << "output: " << std::endl << output << std::endl;
}

double *Conv::forward(const double *data) {
    // reshape input data;
    for(int i = 0; i < _inputChannel; i++){
        _input.emplace_back(Eigen::Map<Mat>(const_cast<double*>(data + i * _inputHeight * _inputWidth), _inputHeight, _inputWidth));
    }

    // for each kernel
    for(int i = 0; i < _outputChannel; i++){
        // for each input channel
        for(int j = 0; j < _inputChannel; j++){
            cross_correlation(_input[j], K[i][j], a[i], _stride, _padding);
        }
        // std::cout << a[i] << std::endl;
        a[i].array() += b(i);
        a[i] = a[i].unaryExpr(std::ref(activation));
    }

    // copy to output
    // eigen use column major
    for(int i = 0; i < _outputChannel; i++){
        std::copy(a[i].data(), a[i].data() + _outputHeight * _outputWidth, _output + i * _outputHeight * _outputWidth);
    }
    return _output;
}

double *Conv::backward(const double *grad) {
    // reshape grad
    Mat3d dz;
    for(int i = 0; i < _outputChannel; i++){
        dz.emplace_back(Eigen::Map<Mat>(const_cast<double*>(grad + i * _outputHeight * _outputWidth), _outputHeight, _outputWidth));
        dz.back().cwiseProduct(a[i].unaryExpr(std::ref(derivative)));
    }
    
    // calculate db
    for(int i = 0; i < _outputChannel; i++){
        db(i) += dz[i].sum();
    }

    // add dilation 
    Mat3d dz_dilation(_outputChannel, Mat::Zero(_outputHeight * _stride - _stride + 1, _outputWidth * _stride - _stride + 1));
    for(int i = 0; i < _outputChannel; i++){
        for(int j = 0; j < _outputHeight; j++){
            for(int k = 0; k < _outputWidth; k++){
                dz_dilation[i](j * _stride, k * _stride) = dz[i](j, k);
            }
        }
    }
    
    // calculate dK
    for(int i = 0; i < _outputChannel; i++){
        for(int j = 0; j < _inputChannel; j++){
            Mat dk_ij = Mat::Zero(_kernel_size, _kernel_size);
            cross_correlation(_input[j], dz_dilation[i], dk_ij, 1, _padding);
            dK[i][j].noalias() += dk_ij;
        }
    }

    // calculate dX
    Mat3d dX(_inputChannel, Mat::Zero(_inputHeight + 2 * _padding, _inputWidth + 2 * _padding));
    for(int i = 0; i < _inputChannel; i++){
        Mat dx_ij = Mat::Zero(_inputHeight + 2 * _padding, _inputWidth + 2 * _padding);
        for(int j = 0; j < _outputChannel; j++){
            correlation(dz_dilation[j], K[j][i], dx_ij, 1, _kernel_size - 1);
            dX[i].noalias() += dx_ij;
        }
    }

    // std::cout << "dX: " << std::endl << dX[0] << std::endl;

    // remove padding and save to 
    for(int i = 0; i < _inputChannel; i++){
        for(int j = 0; j < _inputHeight; j++){
            for(int k = 0; k < _inputWidth; k++){
                std::copy(dX[i].data() + (j + _padding) * (_inputWidth + 2 * _padding) + k + _padding, 
                        dX[i].data() + (j + _padding) * (_inputWidth + 2 * _padding) + k + _padding + 1, 
                        _grad + i * _inputHeight * _inputWidth + j * _inputWidth + k);
            }
        }
    }
    return _grad;
}

void Conv::update(double lr, int batchSize)
{
    double scaler = lr / batchSize;
    for(int i = 0; i < _outputChannel; i++){
        for(int j = 0; j < _inputChannel; j++){
            K[i][j] -= dK[i][j] * scaler;
            dK[i][j].setZero();
        }
    }
    b -= db * scaler;
    db.setZero();
}

void Conv::setKernel(int i, double *kernel)
{
    K[i][0] = Eigen::Map<Mat>(kernel, _kernel_size, _kernel_size);
}
