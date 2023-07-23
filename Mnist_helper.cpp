#include "Mnist_helper.h"

void load(float **data, int *label, float **test_data, int *test_label)
{

	char* data_fn = "../data/train-images-idx3-ubyte";
	char* target_fn = "../data/train-labels-idx1-ubyte";
	char* test_data_fn = "../data/t10k-images-idx3-ubyte";
	char* test_target_fn = "../data/t10k-labels-idx1-ubyte";

	mnist_data *data1, *data2;
	unsigned int cnt1, cnt2;

	mnist_load(data_fn, target_fn, &data1, &cnt1);
	mnist_load(test_data_fn, test_target_fn, &data2, &cnt2);

	if(cnt1 != 60000 || cnt2 != 10000){
		printf("Error: file corrupted\n");
		exit(1);
	}

	for(int i = 0; i < cnt1; i++){
		for(int j = 0; j < 784; j++){
			data[i][j] = data1[i].data[j/28][j%28];
		}
		label[i] = data1[i].label;
	}

	for(int i = 0; i < cnt2; i++){
		for(int j = 0; j < 784; j++){
			test_data[i][j] = data2[i].data[j/28][j%28];
		}
		test_label[i] = data2[i].label;
	}
	free(data1);
	free(data2);
}

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
