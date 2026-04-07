//
// Created by Loginov Nikolay on 31.03.2026.
//
#include "../include/activation/sigmoid_func.hpp"

namespace nn {
Tensor SigmoidFunc::operator()(const Tensor& input) const {
  Tensor output(input.rows(), input.columns());
  output.data() = (1 / (1 + (-input.data().array()).exp())).matrix();
  return output;
}
Tensor SigmoidFunc::derivative(const Tensor& input, const Tensor& output) const {
  (void)input;
  Tensor result(output.rows(), output.columns());
  result.data() = (output.data().array() * (1 - output.data().array())).matrix();
  return result;
}
}  // namespace nn
