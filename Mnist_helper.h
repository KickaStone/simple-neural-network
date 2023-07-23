#ifndef __MNIST_HELPER_H__
#define __MNIST_HELPER_H__

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#define MNIST_STATIC

#include "mnist.h"

void load(float **data, int *label, float **test_data, int *test_label);
int convert(double **data, unsigned int *label, mnist_data *md, int num_data);

#endif // __MNIST_HELPER_H__
