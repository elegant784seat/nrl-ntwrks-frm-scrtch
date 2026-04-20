//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once
#include <any>

#include "../verify/verify.hpp"
#include "Linalg.hpp"
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
    Matrix bias;
  };
  struct ForwardResult {
    std::any state;
    Matrix output;
  };
  struct BackwardResult {
    std::any grad;
    Matrix grad_input;
  };

  LinLayer(In input_dim, Out output_dim, Random& random = GlobalRandom());
  Matrix predict(const Matrix& input) const;

  ForwardResult forward(const Matrix& input) const;

  BackwardResult backward(const std::any& state, const Matrix& grad_output) const;
  void update(const std::any& state, const std::any& grad, std::any& optimizer, std::any& cache);

  Index input_dim() const;
  Index output_dim() const;

  const Matrix& weights() const;
  const RowVector& bias() const;

 private:
  int input_dim_;
  int output_dim_;

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
