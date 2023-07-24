#ifndef __MNIST_HELPER_H__
#define __MNIST_HELPER_H__

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#define MNIST_STATIC

#include "mnist.h"
#include <vector>

void load(float **data, int *label, float **test_data, int *test_label);
int convert(double **data, unsigned int *label, mnist_data *md, int num_data);

void load(std::vector<float*> &data, std::vector<int> &label, std::vector<float*> &test_data, std::vector<int> &test_label);
void load(std::vector<double*> &data, std::vector<int> &label, std::vector<double*> &test_data, std::vector<int> &test_label);

#endif // __MNIST_HELPER_H__



