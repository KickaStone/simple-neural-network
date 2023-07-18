#include "Mnist_helper.h"

int convert(double **data, unsigned int *label, mnist_data *md, int num_data)
{
	for(int i = 0; i < num_data; i++){
		for(int j = 0; j < 784; j++){
			data[i][j] = md[i].data[j/28][j%28];
		}
		label[i] = md[i].label;
	}
	return 0;
}
