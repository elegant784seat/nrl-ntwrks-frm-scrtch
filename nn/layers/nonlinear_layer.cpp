#include "../layers/nonlinear_layer.hpp"

#include "../verify/verify.hpp"

namespace nn {
NonLinLayer::NonLinLayer(AnyFunc func) : func_(std::move(func)) {}

Matrix NonLinLayer::predict(const Matrix& input) const { return func_(input); }

NonLinLayer::ForwardResult NonLinLayer::forward(const Matrix& input) const {
  auto output = func_(input);
  State state{input, output};
  return {std::any(std::move(state)), std::move(output)};
}

NonLinLayer::BackwardResult NonLinLayer::backward(const std::any& state,
                                                  const Matrix& grad_output) const {
  NN_VERIFY(state.has_value());
  const auto& layer_state = std::any_cast<const State&>(state);
  NN_VERIFY(layer_state.input.rows() == grad_output.rows());
  NN_VERIFY(layer_state.input.cols() == grad_output.cols());
  NN_VERIFY(layer_state.output.rows() == grad_output.rows());
  NN_VERIFY(layer_state.output.cols() == grad_output.cols());
  Matrix local_grad = func_.derivative(layer_state.input, layer_state.output);
  Matrix grad_input = (grad_output.array() * local_grad.array()).matrix();
  return {std::any{}, std::move(grad_input)};
}
void NonLinLayer::update(const std::any& state, const std::any& grad,  std::any& optimizer,
                          std::any& cache) {
  (void)state, (void)grad, (void)optimizer, (void)cache;
}

}  // namespace nn
// Created by Loginov Nikolay on 30.03.2026.
//
