//
// Created by Loginov Nikolay on 08.05.2026.
//

#pragma once
#include <cmath>

#include "Linalg.hpp"
#include "verify/verify.hpp"

namespace nn {
class SoftMaxCrossEntropyLoss {
 public:
  Scalar operator()(const Matrix& logits, const Matrix& target) const {
    NN_VERIFY(logits.rows() == target.rows());
    NN_VERIFY(logits.cols() == target.cols());
    NN_VERIFY(logits.rows() > 0);
    NN_VERIFY(logits.cols() > 0);

    auto log_probs = logSoftmax(logits);

    auto loss = -(target.array() * log_probs.array()).sum();
    return loss / static_cast<Scalar>(logits.rows());
  }
  Matrix gradient(const Matrix& logits, const Matrix& target) const {
    NN_VERIFY(logits.rows() == target.rows());
    NN_VERIFY(logits.cols() == target.cols());
    NN_VERIFY(logits.rows() > 0);
    NN_VERIFY(logits.cols() > 0);

    auto probs = softmax(logits);
    return (probs - target) / static_cast<Scalar>(logits.rows());
  }

 private:
  static Matrix softmax(const Matrix& logits) {
    Matrix result(logits.rows(), logits.cols());
    for (Index i = 0; i < logits.rows(); ++i) {
      auto max_val = logits.row(i).maxCoeff();

      result.row(i) = (logits.row(i).array() - max_val).exp();

      auto sum = result.row(i).sum();
      NN_VERIFY(sum > 0);

      result.row(i) = result.row(i) / sum;
    }
    return result;
  }
  static Matrix logSoftmax(const Matrix& logits) {
    Matrix result(logits.rows(), logits.cols());

    for (Index i = 0; i < logits.rows(); ++i) {
      auto max_val = logits.row(i).maxCoeff();
      auto sum_exp = (logits.row(i).array() - max_val).exp().sum();
      NN_VERIFY(sum_exp > 0);

      auto log_sum_exp = max_val + std::log(sum_exp);
      result.row(i) = logits.row(i).array() - log_sum_exp;
    }
    return result;
  }
};
}  // namespace nn
