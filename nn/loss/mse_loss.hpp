//
// Created by Loginov Nikolay on 08.05.2026.
//

#pragma once
#include "Linalg.hpp"
#include "verify/verify.hpp"

namespace nn {
class MseLoss {
 public:
  Scalar operator()(const Matrix& prediction, const Matrix& target) const {
    NN_VERIFY(prediction.rows() == target.rows());
    NN_VERIFY(prediction.cols() == target.cols());
    NN_VERIFY(prediction.size() > 0);

    auto diff = prediction - target;
    return diff.array().square().mean();
  }
  Matrix gradient(const Matrix& prediction, const Matrix& target) const {
    NN_VERIFY(prediction.rows() == target.rows());
    NN_VERIFY(prediction.cols() == target.cols());
    NN_VERIFY(prediction.size() > 0);
    return 2 * (prediction - target) / static_cast<Scalar>(prediction.size());
  }
};
}  // namespace nn
