//#include "Network.h"
#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"


void test(){
    mnist_data *data;
    unsigned int cnt;

    int ret = mnist_load(R"(E:\Projects\CPlusPlusProjects\testcmake\data\train-images-idx3-ubyte)", R"(E:\Projects\CPlusPlusProjects\testcmake\data\train-labels-idx1-ubyte)", &data, &cnt);
    if(ret != 0){
        printf("error\n");
        return;
    }
    
    // print first iamge and label
    for(int i = 0; i < 28; ++i){
        for(int j = 0; j < 28; ++j){
            if(data[0].data[i][j] > 0.1)
                printf("1");
            else
                printf("0");
        }
        printf("\n");
    }

    printf("%d\n", data[0].label);

    double **n_data;
    n_data = new double*[cnt];
    for(int i = 0; i < cnt; ++i){
        n_data[i] = new double[28 * 28];
        for(int j = 0; j < 28; ++j){
            for(int k = 0; k < 28; ++k){
                n_data[i][j * 28 + k] = data[i].data[j][k];
            }
        }
    }

    for(int i = 0; i < 10; i ++){
        //print
        for(int x = 0; x < 28; ++x){
            for(int y = 0; y < 28; ++y){
                if(n_data[i][x * 28 + y] > 0.1)
                    printf("1");
                else
                    printf("0");
            }
            printf("\n");
        }
    }
}

int main(){
    test();
    return 0;
}