//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once

#include "../verify/verify.hpp"
#include "Linalg.hpp"
#include "../optimizer/any_optimizer.hpp"
#include "random.hpp"

namespace nn {

enum In : Index;
enum Out : Index;

class LinLayer {
 public:
  struct State {
    Matrix input;
  };

  struct Grad {
    Matrix weights;
    RowVector bias;
  };
  using Cache = AnyCache;

  struct ForwardResult {
    State state;
    Matrix output;
  };

  struct BackwardResult {
    Grad grad;
    Matrix grad_input;
  };

 public:
  LinLayer(In input_dim, Out output_dim, Random& random = Random::Global());
  Matrix predict(const Matrix& input) const;

  ForwardResult forward(Matrix&& input) const;

  BackwardResult backward(const State& state, const Matrix& grad_output) const;
  void update(const State& state, const Grad& grad, AnyOptimizer& optimizer, Cache& cache);

  Index input_dim() const;
  Index output_dim() const;

  const Matrix& weights() const;
  const RowVector& bias() const;

  Grad zeroGrad() const;

 private:
  Matrix weights_;
  RowVector bias_;
};
LinLayer::Grad& operator+=(LinLayer::Grad& left, const LinLayer::Grad& right);

LinLayer::Grad& operator-=(LinLayer::Grad& left, const LinLayer::Grad& right);

LinLayer::Grad& operator*=(LinLayer::Grad& grad, Scalar scalar);

LinLayer::Grad& operator/=(LinLayer::Grad& grad, Scalar scalar);

LinLayer::Grad operator+(LinLayer::Grad left, const LinLayer::Grad& right);

LinLayer::Grad operator-(LinLayer::Grad left, const LinLayer::Grad& right);

LinLayer::Grad operator*(LinLayer::Grad grad, Scalar scalar);

LinLayer::Grad operator*(Scalar scalar, LinLayer::Grad grad);

LinLayer::Grad operator/(LinLayer::Grad grad, Scalar scalar);
}  // namespace nn
