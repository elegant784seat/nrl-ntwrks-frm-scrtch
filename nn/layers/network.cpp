//
// Created by Loginov Nikolay on 04.05.2026.
//

#include "network.hpp"

namespace nn {
Matrix Network::predict(const Matrix& input) const {
  NN_VERIFY(!layers_.empty());
  Matrix output = input;
  for (const auto& layer : layers_) {
    output = layer.predict(output);
  }
  return output;
}

Network::ForwardResult Network::forward(Matrix&& input) const {
  NN_VERIFY(!layers_.empty());
  State state;
  state.layer_states.reserve(layers_.size());
  Matrix cur = std::move(input);
  for (const auto& layer : layers_) {
    auto result = layer.forward(std::move(cur));
    state.layer_states.push_back(std::move(result.state));
    cur = std::move(result.output);
  }
  return ForwardResult{.state = std::move(state), .output = std::move(cur)};
}

Network::BackwardResult Network::backward(const State& state, const Matrix& grad_output) const {
  NN_VERIFY(!layers_.empty());
  NN_VERIFY(state.layer_states.size() == layers_.size());

  Grad grad;
  grad.layer_grads.resize(layers_.size());
  Matrix cur_grad = grad_output;
  for (size_t i = layers_.size(); i > 0; --i) {
    size_t id_layer = i - 1;
    auto result = layers_[id_layer].backward(state.layer_states[id_layer], cur_grad);
    grad.layer_grads[id_layer] = std::move(result.grad);
    cur_grad = std::move(result.grad_input);
  }
  return BackwardResult{.grad = std::move(grad), .grad_input = std::move(cur_grad)};
}

void Network::update(const State& state, const Grad& grad, AnyOptimizer& optimizer, Cache& cache) {
  NN_VERIFY(!layers_.empty());
  NN_VERIFY(state.layer_states.size() == layers_.size());
  NN_VERIFY(grad.layer_grads.size() == layers_.size());
  NN_VERIFY(cache.layer_caches.size() == layers_.size());
  for (size_t i = 0; i < layers_.size(); ++i) {
    layers_[i].update(state.layer_states[i], grad.layer_grads[i], optimizer, cache.layer_caches[i]);
  }
}

Network::Cache Network::initCache(const AnyOptimizer& optimizer) const {
  Cache cache;
  cache.layer_caches.reserve(layers_.size());
  for (const auto& layer : layers_) {
    cache.layer_caches.push_back(layer.initCache(optimizer));
  }
  return cache;
}
}  // namespace nn
