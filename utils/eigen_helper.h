#ifndef EIGEN_HELPER_H
#define EIGEN_HELPER_H

#include <Eigen/Dense>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using VecMap = Eigen::Map<Vec>;
using MatMap = Eigen::Map<Mat>;

static void reshape_matrix(const double* input, Mat &mat, int rows, int cols){
    mat = MatMap(const_cast<double*>(input), rows, cols);
}

static void reshape_vector(const double* input, Vec &vec, int size){
    vec = VecMap(const_cast<double*>(input), size);
}

static void change_pointer_matrix(MatMap &mat, double* &ptr, int rows, int cols){
    new (&mat) MatMap(ptr, mat.rows() ? mat.rows() : rows, mat.cols() ? mat.cols() : cols);
}

static void change_pointer_vector(VecMap &vec, double* &ptr, int size){
    new (&vec) VecMap(ptr, vec.size() ? vec.size() : size);
}

static void reshape_to_ptr_m(const Mat &mat, double* &ptr){
    ptr = const_cast<double*>(mat.data());
}

static void copy_to_ptr_m(const Mat &mat, double* &ptr){
    ptr = new double[mat.size()];
    std::copy(mat.data(), mat.data() + mat.size(), ptr);
}

static void reshape_to_ptr_v(const Vec &vec, double* &ptr){
    ptr = const_cast<double*>(vec.data());
}

#endif //EIGEN_HELPER_H