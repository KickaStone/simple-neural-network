#include "layer.h"

class Dense : public Layer{
    // public: 
private:
        double *w;
        double *b;
        double *a;
        double *dw;
        double *db;
        double *dz;

        const double *input;
    
    public:
        Dense(int input_size, int output_size, Activation::ActivationFunctionType TYPE);

        ~Dense() override;
        double* forward(const double *input_data) override;
        double* backward(const double *grad) override;
        void update(double lr, int batchSize) override;

        [[nodiscard]] double* getW() const { return w; }
        [[nodiscard]] double* getB() const { return b; }
        [[nodiscard]] double* getA() const { return a; }
        [[nodiscard]] double* getDw() const { return dw; }
        [[nodiscard]] double* getDb() const { return db; }
        [[nodiscard]] double* getDz() const { return dz; }

        void setW(double *ww) { memcpy(this->w, ww, sizeof(double) * inputSize * outputSize);}
        void setB(double *bb) { memcpy(this->b, bb, sizeof(double) * outputSize);}
        void setA(double *aa) { memcpy(this->a, aa, sizeof(double) * outputSize);}
        void setInput(double* input) { this->input = input; }

};
