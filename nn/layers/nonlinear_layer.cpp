#include "../layers/nonlinear_layer.hpp"

#include "../verify/verify.hpp"

namespace nn {
NonLinLayer::NonLinLayer(AnyFunc func) : func_(std::move(func)) {}

Matrix NonLinLayer::predict(const Matrix& input) const { return func_(input); }

NonLinLayer::ForwardResult NonLinLayer::forward(Matrix&& input) const {
  Matrix output = func_(input);
  return ForwardResult{.state = State{.input = std::move(input), .output = output},
                       .output = std::move(output)};
}

NonLinLayer::BackwardResult NonLinLayer::backward(const State& state,
                                                  const Matrix& grad_output) const {
  NN_VERIFY(state.input.rows() == grad_output.rows());
  NN_VERIFY(state.input.cols() == grad_output.cols());
  NN_VERIFY(state.output.rows() == grad_output.rows());
  NN_VERIFY(state.output.cols() == grad_output.cols());
  Matrix local_grad = func_.derivative(state.input, state.output);
  NN_VERIFY(local_grad.rows() == grad_output.rows());
  NN_VERIFY(local_grad.cols() == grad_output.cols());
  Matrix grad_input = (grad_output.array() * local_grad.array()).matrix();
  return BackwardResult{.grad = Grad{}, .grad_input = std::move(grad_input)};
}
void NonLinLayer::update(const State& state, const Grad& grad, AnyOptimizer& optimizer,
                         Cache& cache) {
  (void)state, (void)grad, (void)optimizer, (void)cache;
}

NonLinLayer::Grad NonLinLayer::zeroGrad() const { return Grad{}; }

}  // namespace nn
// Created by Loginov Nikolay on 30.03.2026.
//
