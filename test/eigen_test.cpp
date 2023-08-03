//
// Created by JunchengJi on 8/1/2023.
//
#include <Eigen/Dense>
#include <iostream>
#include <gtest/gtest.h>

using namespace std;
using namespace Eigen;

TEST(eigen, test1){
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    m.block(0,0,1,2) = Eigen::MatrixXd::Zero(1,2);
    std::cout << m << std::endl;
}

TEST(eigen, test2){
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    m.block(0,0,1,2) = Eigen::MatrixXd::Zero(1,2);
    std::cout << m << std::endl;
}

TEST(eigen, test3){
    Eigen::MatrixXd a(2,2);
    Eigen::MatrixXd b(a.rows() + 2, a.cols() + 2);
    b.setZero();
    a << 1, 2, 3, 4;
    b.block(1,1,2,2) = a;
    cout << b << endl;
}

TEST(eigen, test4){
    Eigen::MatrixXd a{{1,2},{3,4}};
    a.transposeInPlace();
    cout << a << endl;

    Eigen::MatrixXd b(2,3);
    b << 1,2,3,4,5,6;
    b.transposeInPlace();
    cout << b << endl;
}

TEST(eigen, ArrayMul){
    Eigen::ArrayXXd a {{1,2},{3,4}};
    Eigen::ArrayXXd b {{1,2},{3,4}};
    Eigen::ArrayXXd c;
    cout << "a = " << endl << a << endl << endl;
    cout << "b = " << endl << b << endl << endl;
    cout << "a * b = " << endl << a * b << endl << endl;
    // c = a * b;
    // cout << c << endl;
}

TEST(eigen, ArrayOp){
    Eigen::Array22d a = Eigen::Array22d::Random();
    a *= 2;
    cout << "a = " << endl << a << endl << endl;
    cout << "a.abs() = " << endl << a.abs() << endl << endl;
    cout << "a.abs().sqrt() = " << endl << a.abs().sqrt() << endl << endl;
    cout << "a.min(a.abs().sqrt()) = " << endl << a.min(a.abs().sqrt()) << endl << endl;
}

TEST(eigen, convert){
    Eigen::Matrix2d a {{1,2},{3,4}};
    Eigen::MatrixXd b {{1,2},{3,4}};

    // cout << a.cwiseProduct(b) << endl;
    // cout << a.array() * b.array() << endl;

    Eigen::Array22d c{{1,2},{3,4}};
    Eigen::Array22d d{{1,2},{3,4}};
    cout << c.matrix() * d.matrix() << endl;
    cout << c.cwiseProduct(d) << endl;

    MatrixXd e;
    e = c * d;
    cout << e << endl;

    ArrayXXd f;
    f = a * b;
    cout << f << endl;
}

TEST(eigen, block){
    Eigen::MatrixXd a(2,2);
    Eigen::MatrixXd b(4,4);
    a << 1,2,3,4;
    b.setZero();
    b.block(1,1,2,2) = a; 
    // equivalent with  b.block<2,2>(1,1) = a; 
    // start from (1,1) with size (2,2)
    cout << b << endl;
}

TEST(eigen, slicingAndIndexing){
    Eigen::MatrixXd a(4, 4);
    a.setConstant(4);

    Eigen::VectorXd b(10);
    b.setConstant(10);
    b(last-fix<7>, last-fix<2>);
    b(last-7, fix<6>);
}

TEST(eigen, slicing){
    MatrixXd m = MatrixXd::Random(6, 6);
    cout << "m = " << endl << m << endl << endl;
    std::vector<int> indices = {4, 3, 1, 2, 2};
    cout << "m(indices, indices) = " << endl << m(indices, indices) << endl << endl;
}

TEST(eigen, slicing2){
    MatrixXd a(2, 3);
    a.reshaped() = VectorXd::LinSpaced(6, 0, 5);
    cout << "a = " << endl << a << endl << endl;
}

TEST(eigen, init){
    MatrixXd a(2, 3);
    a = MatrixXd::Constant(2, 3, 1);
    cout << "a = " << endl << a << endl << endl;

    MatrixXd b = MatrixXd::Random(3,3);
    MatrixXd c = b * MatrixXd::Identity(3,3);
    cout << "c = " << endl << c << endl << endl;
    MatrixXd d = (MatrixXd(2, 2) << 2, 2, 2, 2).finished() * a;
    cout << "d = " << endl << d << endl << endl;
}

TEST(eigen, reduction){
    MatrixXd a(2, 3);
    a = MatrixXd::Constant(2, 3, 1);
    cout << "a = " << endl << a << endl << endl;
    cout << "a.sum() = " << endl << a.sum() << endl << endl;
    cout << "a.colwise().sum() = " << endl << a.colwise().sum() << endl << endl;
    cout << "a.rowwise().sum() = " << endl << a.rowwise().sum() << endl << endl;
    // const auto func = [](int c){return c > 2;};
    // cout << a.redux(func) << endl;
}


TEST(eigen, visiter){
    MatrixXd a(2, 2);
    a << 1, 2, 3, 4;

    Eigen::Index maxRow, maxCol;
    float max = a.maxCoeff(&maxRow, &maxCol);
    cout << "max = " << max << endl;
    cout << "maxRow = " << maxRow << endl;
    cout << "maxCol = " << maxCol << endl;
}

TEST(eigen, partialReduction){
    MatrixXd a(2, 3);
    a = MatrixXd::Random(2, 3);
    cout << "a = " << endl << a << endl << endl;
    cout << "a.colwise().sum() = " << endl << a.colwise().sum() << endl << endl;
    cout << "a.colwise().sum().maxCoeff() = " << endl << a.colwise().sum().maxCoeff() << endl << endl;
}


TEST(eigen, blockOperation){
    MatrixXd a = MatrixXd::Constant(6, 6, 1);
    cout << "a = " << endl << a << endl << endl;
    a(Eigen::seqN(0, 2), Eigen::seqN(0, 2)).array() += 1;
    cout << "a = " << endl << a << endl << endl;
}

TEST(eigen, blockOp2){
    MatrixXd a = MatrixXd::Constant(6, 6, 1);
    cout << "a = " << endl << a << endl << endl;
    // cout << a.block(4, 4, 3, 3);
    // cout << a.block(-1, -1, 2, 2); // error
    cout << a.block<7, 7>(0, 0);
}
