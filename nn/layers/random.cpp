//
// Created by Loginov Nikolay on 20.04.2026.
//
#include "random.hpp"

namespace nn {

Matrix Random::normalMatrix(Index rows, Index cols, Scalar mean, Scalar stddev) {
  return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stddev);
}

RowVector Random::normalRowVector(Index size, Scalar mean, Scalar stddev) {
  return Eigen::Rand::normal<RowVector>(1, size, generator_, mean, stddev);
}

Random& GlobalRandom() {
  static Random random;
  return random;
}

}  // namespace nn
