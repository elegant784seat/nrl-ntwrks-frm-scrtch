//
// Created by Loginov Nikolay on 20.04.2026.
//
#include "random.hpp"

namespace nn {

Random& Random::Global() {
  static Random random;
  return random;
}

Matrix Random::normalMatrix(Index rows, Index cols, Scalar mean, Scalar stddev) {
  return Eigen::Rand::normal<Matrix>(rows, cols, generator_, mean, stddev);
}

RowVector Random::normalRowVector(Index size, Scalar mean, Scalar stddev) {
  return Eigen::Rand::normal<RowVector>(1, size, generator_, mean, stddev);
}

void Random::shuffle(std::vector<Index>& indices) {
  std::shuffle(indices.begin(), indices.end(), generator_);
}

}  // namespace nn
