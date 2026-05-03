//
// Created by Loginov Nikolay on 29.03.2026.
//
#pragma once

#include <any>

#include "../activation/any_func.hpp"
#include "../verify/verify.hpp"
#include "Linalg.hpp"

namespace nn {

class NonLinLayer {
  struct State {
    Matrix input;
    Matrix output;
  };
  struct ForwardResult {
    std::any state;
    Matrix output;
  };
  struct BackwardResult {
    std::any grad;
    Matrix grad_input;
  };

 public:
  explicit NonLinLayer(AnyFunc func);

  Matrix predict(const Matrix& input) const;

  ForwardResult forward(Matrix&& input) const;

  BackwardResult backward(const std::any& state, const Matrix& grad_output) const;
  void update(const std::any& state, const std::any& grad, std::any& optimizer, std::any& cache);

 private:
  AnyFunc func_;
};
}  // namespace nn
