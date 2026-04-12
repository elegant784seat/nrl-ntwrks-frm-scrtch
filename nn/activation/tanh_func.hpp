//
// Created by Loginov Nikolay on 31.03.2026.
//

#pragma once
#include "../tensor.hpp"

namespace nn {
class TanhFunc {
 public:
  Tensor operator()(const Tensor& input) const;
  Tensor derivative(const Tensor& input, const Tensor& output) const;
};
}  // namespace nn
