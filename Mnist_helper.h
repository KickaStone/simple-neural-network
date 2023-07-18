#ifndef __MNIST_HELPER_H__
#define __MNIST_HELPER_H__

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#define MNIST_STATIC

#include "mnist.h"

int convert(double **data, unsigned int *label, mnist_data *md, int num_data);

#endif // __MNIST_HELPER_H__
