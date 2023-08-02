#include "layer.h"


class Dense : public Layer{
    // public: 
private:
    // double *w;
    // double *b;
    // double *a;
    // double *dw;
    // double *db;
    // double *dz;
    // double *input_grad;
    // const double *input;
    Mat w;
    Vec a;
    Vec b;
    Mat dw;
    Vec db;
    Vec dz;
    Vec input_grad;
    const double *input;

    
public:
    Dense(int input_size, int output_size, Activation::ActivationFunctionType TYPE);

    ~Dense() override;
    double* forward(const double *input_data) override;
    double* backward(const double *grad) override;
    void update(double lr, int batchSize) override;
};
