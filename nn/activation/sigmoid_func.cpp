//
// Created by Loginov Nikolay on 31.03.2026.
//
#include "../activation/sigmoid_func.hpp"

namespace nn {
Matrix SigmoidFunc::operator()(const Matrix& input) const {
  return (1 / (1 + (-input.array()).exp())).matrix();
}
Matrix SigmoidFunc::derivative(const Matrix& input, const Matrix& output) const {
  (void)input;
  return (output.array() * (1 - output.array())).matrix();
}
}  // namespace nn
