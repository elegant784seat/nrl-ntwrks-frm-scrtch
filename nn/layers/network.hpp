//
// Created by Loginov Nikolay on 04.05.2026.
//

#pragma once
#include <any>

#include "Linalg.hpp"
#include "any_layer.hpp"

namespace nn {
class Network {
 public:
  struct State {
    std::vector<AnyState> layer_states;
  };

  struct Grad {
    std::vector<AnyGrad> layer_grads;
  };

  struct Cache {
    std::vector<AnyCache> layer_caches;
  };

  struct ForwardResult {
    State state;
    Matrix output;
  };

  struct BackwardResult {
    Grad grad;
    Matrix grad_input;
  };

  Network() = default;

  explicit Network(std::vector<AnyLayer> layers) : layers_(std::move(layers)) {
    NN_VERIFY(!layers_.empty());
  }

  void add(AnyLayer layer) { layers_.push_back(std::move(layer)); }

  Matrix predict(const Matrix& input) const;

  ForwardResult forward(Matrix&& input) const;

  BackwardResult backward(const State& state, const Matrix& grad_output) const;

  void update(const State& state, const Grad& grad, AnyOptimizer& optimizer, Cache& cache);

  Cache initCache(const AnyOptimizer& optimizer) const;

 private:
  std::vector<AnyLayer> layers_;
};
}  // namespace nn
