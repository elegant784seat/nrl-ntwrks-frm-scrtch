#include "../include/layers/nonlinear_layer.hpp"

#include "verify/verify.hpp"

namespace nn {
NonLinLayer::NonLinLayer(AnyFunc func) : func_(std::move(func)) {}

Tensor NonLinLayer::predict(const Tensor& input) const { return func_(input); }

std::pair<std::any, Tensor> NonLinLayer::forward(const Tensor& input) const {
  Tensor output = func_(input);
  NonLinState state{input, output};
  return {std::any(std::move(state)), std::move(output)};
}

std::pair<std::any, Tensor> NonLinLayer::backward(const std::any& state,
                                                  const Tensor& grad_output) const {
  NN_VERIFY(state.has_value());
  const auto& layer_state = std::any_cast<const NonLinState&>(state);
  NN_VERIFY(layer_state.input.rows() == grad_output.rows());
  NN_VERIFY(layer_state.input.columns() == grad_output.columns());
  NN_VERIFY(layer_state.output.rows() == grad_output.rows());
  NN_VERIFY(layer_state.output.columns() == grad_output.columns());
  Tensor local_grad = func_.derivative(layer_state.input, layer_state.output);
  Tensor grad_input(grad_output.rows(), grad_output.columns());
  grad_input.data() = grad_output.data().array() * local_grad.data().array();
  return {std::any{}, std::move(grad_input)};
}
void NonLinLayer::update(const std::any& state, const std::any& grad,  std::any& optimizer,
                          std::any& cache) {
  (void)state, (void)grad, (void)optimizer, (void)cache;
}

}  // namespace nn
// Created by Loginov Nikolay on 30.03.2026.
//
