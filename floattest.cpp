#include <iostream>
#include <vector>
#include <cmath> 

using namespace std;

int main(int argc, char const *argv[])
{
    float a = 0.05f;
    float b = 0.1f;
    vector<float> w1 = {0.15f, 0.20f, 0.25f, 0.30f};
    vector<float> w2 = {0.40f, 0.45f, 0.50f, 0.55f};
    float b1 = 0.35f;
    float b2 = 0.60f;

    float z1 = w1[0] * a + w1[1] * b + b1;
    float z2 = w1[2] * a + w1[3] * b + b1;

    float a1 = 1.0f / (1.0f + exp(-z1));
    float a2 = 1.0f / (1.0f + exp(-z2));

    printf("%.9lf\n", a1);
    printf("%.9lf\n", a2);

    float z3 = w2[0] * a1 + w2[1] * a2 + b2;
    float z4 = w2[2] * a1 + w2[3] * a2 + b2;

    float a3 = 1.0f / (1.0f + exp(-z3));
    float a4 = 1.0f / (1.0f + exp(-z4));

    printf("%.9lf\n", a3);
    printf("%.9lf\n", a4);

    float delta1 = (a3 - 0.01f) * a3 * (1.0f - a3);
    float delta2 = (a4 - 0.99f) * a4 * (1.0f - a4);

    printf("%.9lf\n", delta1);
    printf("%.9lf\n", delta2);

    float dw5 = delta1 * a1;
    float dw6 = delta1 * a2;
    float dw7 = delta2 * a1;
    float dw8 = delta2 * a2;

    printf("dw5: %.9lf\n", dw5);
    printf("dw6: %.9lf\n", dw6);
    printf("dw7: %.9lf\n", dw7);
    printf("dw8: %.9lf\n", dw8);

    float db3 = delta1;
    float db4 = delta2;

    float delta3 = w2[0] * delta1 + w2[2] * delta2;
    delta3 *= a1 * (1.0f - a1);
    float delta4 = w2[1] * delta1 + w2[3] * delta2;
    delta4 *= a2 * (1.0f - a2);

    printf("delta3: %.9lf\n", delta3);
    printf("delta4: %.9lf\n", delta4);

    float dw1 = delta3 * a;
    float dw2 = delta3 * b;
    float dw3 = delta4 * a;
    float dw4 = delta4 * b;


    printf("dw1: %.9lf\n", dw1);
    printf("dw2: %.9lf\n", dw2);
    printf("dw3: %.9lf\n", dw3);
    printf("dw4: %.9lf\n", dw4);

    float n_w1_0 = w1[0] - 0.5f * dw1;
    float n_w1_1 = w1[1] - 0.5f * dw2;
    float n_w1_2 = w1[2] - 0.5f * dw3;
    float n_w1_3 = w1[3] - 0.5f * dw4;

    float n_w2_0 = w2[0] - 0.5f * dw5;
    float n_w2_1 = w2[1] - 0.5f * dw6;
    float n_w2_2 = w2[2] - 0.5f * dw7;
    float n_w2_3 = w2[3] - 0.5f * dw8;

    printf("n_w1_0: %.9lf\n", n_w1_0);
    printf("n_w1_1: %.9lf\n", n_w1_1);
    printf("n_w1_2: %.9lf\n", n_w1_2);
    printf("n_w1_3: %.9lf\n", n_w1_3);

    printf("n_w2_0: %.9lf\n", n_w2_0);
    printf("n_w2_1: %.9lf\n", n_w2_1);
    printf("n_w2_2: %.9lf\n", n_w2_2);
    printf("n_w2_3: %.9lf\n", n_w2_3);
    

    return 0;
}
