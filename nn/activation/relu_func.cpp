#include "../activation/relu_func.hpp"
namespace nn {

Matrix ReluFunc::operator()(const Matrix& input) const {

  return input.cwiseMax(0);
}

Matrix ReluFunc::derivative(const Matrix& input, const Matrix&) const {
  Matrix result(input.rows(), input.cols());
  return (input.array() > 0).template cast<Scalar>().matrix();
}
}  // namespace nn
// Created by Loginov Nikolay on 31.03.2026.
//
