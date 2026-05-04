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
  struct State {
    Matrix input;
  };
  struct Grad {
    Matrix weights;
    RowVector bias;
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
  LinLayer(In input_dim, Out output_dim, Random& random = Random::Global());
  Matrix predict(const Matrix& input) const;

  ForwardResult forward(Matrix&& input) const;

  BackwardResult backward(const std::any& state, const Matrix& grad_output) const;
  void update(const std::any& state, const std::any& grad, std::any& optimizer, std::any& cache);

  Index input_dim() const;
  Index output_dim() const;

  const Matrix& weights() const;
  const RowVector& bias() const;

 private:
  Matrix weights_;
  RowVector bias_;
};
// LinLayer::Grad& operator+=(LinLayer::Grad& left, const LinLayer::Grad& right);
//
// LinLayer::Grad& operator-=(LinLayer::Grad& left, const LinLayer::Grad& right);
//
// LinLayer::Grad& operator*=(LinLayer::Grad& grad, Scalar scalar);
//
// LinLayer::Grad& operator/=(LinLayer::Grad& grad, Scalar scalar);
//
// LinLayer::Grad operator+(LinLayer::Grad left, const LinLayer::Grad& right);
//
// LinLayer::Grad operator-(LinLayer::Grad left, const LinLayer::Grad& right);
//
// LinLayer::Grad operator*(LinLayer::Grad grad, Scalar scalar);
//
// LinLayer::Grad operator*(Scalar scalar, LinLayer::Grad grad);
//
// LinLayer::Grad operator/(LinLayer::Grad grad, Scalar scalar);
}  // namespace nn
