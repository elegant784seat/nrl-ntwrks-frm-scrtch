//
// Created by Loginov Nikolay on 29.03.2026.
//
#pragma once

#include <any>

#include "../activation/any_func.hpp"
#include "../verify/verify.hpp"
#include "Linalg.hpp"
#include "optimizer/any_optimizer.hpp"

namespace nn {

class NonLinLayer {
 public:
  struct State {
    Matrix input;
    Matrix output;
  };
  struct Grad {};
  struct Cache {};
  struct ForwardResult {
    State state;
    Matrix output;
  };
  struct BackwardResult {
    Grad grad;
    Matrix grad_input;
  };

 public:
  explicit NonLinLayer(AnyFunc func);

  Matrix predict(const Matrix& input) const;

  ForwardResult forward(Matrix&& input) const;

  BackwardResult backward(const State& state, const Matrix& grad_output) const;
  void update(const State& state, const Grad& grad, AnyOptimizer& optimizer, Cache& cache);

  Grad zeroGrad() const;

  Cache initCache(const AnyOptimizer& optimizer) const;

 private:
  AnyFunc func_;
};
}  // namespace nn
