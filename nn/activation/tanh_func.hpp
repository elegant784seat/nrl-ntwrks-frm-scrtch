//
// Created by Loginov Nikolay on 31.03.2026.
//

#pragma once
#include "Linalg.hpp"

namespace nn {
class TanhFunc {
 public:
  Matrix operator()(const Matrix& input) const;
  Matrix derivative(const Matrix& input, const Matrix& output) const;
};
}  // namespace nn
