#include "../activation/tanh_func.hpp"

namespace nn {
Matrix TanhFunc::operator()(const Matrix& input) const { return input.array().tanh().matrix(); }

Matrix TanhFunc::derivative(const Matrix& input, const Matrix& output) const {
  (void)input;
  return (1 - output.array().square()).matrix();
}
}  // namespace nn
// Created by Loginov Nikolay on 31.03.2026.
//
