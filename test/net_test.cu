#define DEBUG

#include <gtest/gtest.h>
#include <vector>

#include "../network.cuh"


std::vector<int> layers = {2, 2};
NeuralNetwork nn = NeuralNetwork(2, layers, 0.5);
double* input = new double[2]{0.05f, 0.1f};
double* y = new double[2]{0.01f, 0.99f};

TEST(NetworkTest, init){
    nn.setParams(0.5, 1);
    std::vector<double*> w, b;
    spdlog::set_level(spdlog::level::debug);
    double w1[] = {0.15f, 0.2f, 0.25f, 0.30f};
    double w2[] = {0.4f, 0.45f, 0.5f, 0.55f};
    double b1[] = {0.35f, 0.35f};
    double b2[] = {0.6f, 0.6f};
    w.push_back(w1);
    w.push_back(w2);
    b.push_back(b1);
    b.push_back(b2);
    nn._debug_set(w, b);
}

TEST(NetworkTest, checkParamsSize){
    spdlog::debug("checkParamsSize");
    Params p = nn._debug_params();
    EXPECT_EQ(p.num_layers, 2);
    EXPECT_EQ(p.layers[0], 2);
    EXPECT_EQ(p.layers[1], 2);
    EXPECT_EQ(p.inputsize, 2);
    EXPECT_EQ(p.outputsize, 2);
}

TEST(NetworkTest, checkwb){
    std::vector<double*> w1(2);
    std::vector<double*> b1(2);
    for(int i = 0; i < 2; i++){
        w1[i] = new double[4];
        b1[i] = new double[2];
    }
    nn._debug_get_weights_and_biases(w1, b1);
    EXPECT_TRUE(fabs(w1[0][0] - 0.15f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[0][1] - 0.2f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[0][2] - 0.25f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[0][3] - 0.3f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[1][0] - 0.4f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[1][1] - 0.45f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[1][2] - 0.5f) < 0.00001f);
    EXPECT_TRUE(fabs(w1[1][3] - 0.55f) < 0.00001f);
    EXPECT_TRUE(fabs(b1[0][0] - 0.35f) < 0.00001f);
    EXPECT_TRUE(fabs(b1[0][1] - 0.35f) < 0.00001f);
}

TEST(NetworkTest, forward){
    double *output = nn.forward(input, 2);
    std::vector<double *> a(2);
    for(int i = 0; i < 2; i++){
        a[i] = new double[2];
    }
    nn._debug_get_a(a);
    EXPECT_TRUE(fabs(a[0][0] - 0.59326999f) < 0.00001f);
    EXPECT_TRUE(fabs(a[0][1] - 0.59688438f) < 0.00001f);
    EXPECT_TRUE(fabs(a[1][0] - 0.75136507f) < 0.00001f);
    EXPECT_TRUE(fabs(a[1][1] - 0.772928465f) < 0.00001f);

    // double loss = 0;
    // loss = nn.getLoss(y);
    // spdlog::debug("loss: {}", loss);

    // EXPECT_TRUE(fabs(output[0] - 0.75136507f) < 0.00001f);
    // EXPECT_TRUE(fabs(output[1] - 0.772928465f) < 0.00001f);
}

TEST(NetworkTest, backprop){
    nn.backprop(y);
    std::vector<double*> dw(2);
    std::vector<double*> db(2);

    std::vector<double*> delta(2);
    for(int i = 0; i < 2; i++){
        dw[i] = new double[4];
        db[i] = new double[2];
        delta[i] = new double[2];
    }
    nn._debug_get_grad(dw, db);
    nn._debug_get_delta(delta);
    //print delta
    spdlog::debug("delta[0][0]: {}", delta[0][0]);
    spdlog::debug("delta[0][1]: {}", delta[0][1]);
    spdlog::debug("delta[1][0]: {}", delta[1][0]);
    spdlog::debug("delta[1][1]: {}", delta[1][1]);


    spdlog::debug("dw[0][0]: {}", dw[0][0]);
    spdlog::debug("dw[0][1]: {}", dw[0][1]);
    spdlog::debug("dw[0][2]: {}", dw[0][2]);
    spdlog::debug("dw[0][3]: {}", dw[0][3]);
    spdlog::debug("dw[1][0]: {}", dw[1][0]);
    spdlog::debug("dw[1][1]: {}", dw[1][1]);
    spdlog::debug("dw[1][2]: {}", dw[1][2]);
    spdlog::debug("dw[1][3]: {}", dw[1][3]);
    spdlog::debug("db[0][0]: {}", db[0][0]);
    spdlog::debug("db[0][1]: {}", db[0][1]);
    spdlog::debug("db[1][0]: {}", db[1][0]);
    spdlog::debug("db[1][1]: {}", db[1][1]);
}


TEST(NetworkTest, updateTest){
    nn.update_weights_and_biases();
    std::vector<double*> w(2);
    std::vector<double*> b(2);

    for(int i = 0; i < 2; i++){
        w[i] = new double[4];
        b[i] = new double[2];
    }

    nn._debug_get_weights_and_biases(w, b);
    // print
    spdlog::debug("w[0][0]: {}", w[0][0]);
    spdlog::debug("w[0][1]: {}", w[0][1]);
    spdlog::debug("w[0][2]: {}", w[0][2]);
    spdlog::debug("w[0][3]: {}", w[0][3]);
    spdlog::debug("w[1][0]: {}", w[1][0]);
    spdlog::debug("w[1][1]: {}", w[1][1]);
    spdlog::debug("w[1][2]: {}", w[1][2]);
    spdlog::debug("w[1][3]: {}", w[1][3]);
    spdlog::debug("b[0][0]: {}", b[0][0]);
    spdlog::debug("b[0][1]: {}", b[0][1]);
    spdlog::debug("b[1][0]: {}", b[1][0]);
    spdlog::debug("b[1][1]: {}", b[1][1]);

    EXPECT_TRUE(fabs(w[0][0] - 0.149780716f) < 0.00001f);
    EXPECT_TRUE(fabs(w[0][1] - 0.19956143f) < 0.00001f);
    EXPECT_TRUE(fabs(w[0][2] - 0.24975114f) < 0.00001f);
    EXPECT_TRUE(fabs(w[0][3] - 0.29950229f) < 0.00001f);

    EXPECT_TRUE(fabs(w[1][0] - 0.35891648f) < 0.00001f);
    EXPECT_TRUE(fabs(w[1][1] - 0.408666186f) < 0.00001f);
    EXPECT_TRUE(fabs(w[1][2] - 0.511301270f) < 0.00001f);
    EXPECT_TRUE(fabs(w[1][3] - 0.561370121f) < 0.00001f);

    EXPECT_TRUE(fabs(b[0][0] - 0.34561434f) < 0.00001f);
    EXPECT_TRUE(fabs(b[0][1] - 0.34502287f) < 0.00001f);
    EXPECT_TRUE(fabs(b[1][0] - 0.53075072f) < 0.00001f);
    EXPECT_TRUE(fabs(b[1][1] - 0.61904912f) < 0.00001f);

}

