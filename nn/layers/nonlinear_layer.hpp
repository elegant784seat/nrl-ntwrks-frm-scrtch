//
// Created by Loginov Nikolay on 29.03.2026.
//
#pragma once

#include <any>

#include "../activation/any_func.hpp"

#include "../verify/verify.hpp"
#include "Linalg.hpp"

namespace nn {

struct NonLinState {
  Matrix input;
  Matrix output;
};

class NonLinLayer {
 public:
  explicit NonLinLayer(AnyFunc func);

  Matrix predict(const Matrix& input) const;

  std::pair<std::any, Matrix> forward(const Matrix& input) const;

  std::pair<std::any, Matrix> backward(const std::any& state, const Matrix& grad_output) const;
  void update(const std::any& state, const std::any& grad, std::any& optimizer,  std::any& cache);

 private:
  AnyFunc func_;
};
}  // namespace nn
