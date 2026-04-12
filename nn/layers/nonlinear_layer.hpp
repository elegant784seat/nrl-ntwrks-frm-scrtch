//
// Created by Loginov Nikolay on 29.03.2026.
//
#pragma once

#include <any>

#include "../activation/any_func.hpp"
#include "../tensor.hpp"
#include "../verify/verify.hpp"

namespace nn {

struct NonLinState {
  Tensor input;
  Tensor output;
};

class NonLinLayer {
 public:
  explicit NonLinLayer(AnyFunc func);

  Tensor predict(const Tensor& input) const;

  std::pair<std::any, Tensor> forward(const Tensor& input) const;

  std::pair<std::any, Tensor> backward(const std::any& state, const Tensor& grad_output) const;
  void update(const std::any& state, const std::any& grad, std::any& optimizer,  std::any& cache);

 private:
  AnyFunc func_;
};
}  // namespace nn
