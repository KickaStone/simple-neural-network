#ifndef EIGEN_HELPER_H
#define EIGEN_HELPER_H

#include <Eigen/Dense>
#include <vector>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using VecMap = Eigen::Map<const Vec>;
using MatMap = Eigen::Map<const Mat>;

#endif //EIGEN_HELPER_H