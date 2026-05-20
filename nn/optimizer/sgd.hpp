//
// Created by Loginov Nikolay on 18.05.2026.
//

#pragma once

#include "Linalg.hpp"
#include "any_types.hpp"
#include "layers/linear_layer.hpp"

namespace nn {
class Sgd {
 public:
  explicit Sgd(Scalar learning_rate) : learning_rate_(learning_rate) {
    NN_VERIFY(learning_rate >= 0);
  }
  AnyGrad make(const AnyGrad& grad, AnyCache&) const {
    return AnyGrad(learning_rate_ * grad.get<LinLayer::Grad>());
  }
  AnyCache initCache(const AnyGrad&) const { return AnyCache{}; }

 private:
  Scalar learning_rate_;
};
}  // namespace nn
