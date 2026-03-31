#include "../include/activation/tanh_func.hpp"

namespace nn {
Tensor TanhFunc::operator()(const Tensor& input) const {
  Tensor output(input.rows(), input.columns());
  output.data() = input.data().array().tanh().matrix();
  return output;
}

Tensor TanhFunc::derivative(const Tensor& input, const Tensor& output) const {
  (void) input;
  Tensor result(output.rows(), output.columns());
  result.data() = (1 - output.data().array().square()).matrix();
  return result;
}
} // namespace nn
// Created by Loginov Nikolay on 31.03.2026.
//
