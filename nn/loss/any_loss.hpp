//
// Created by Loginov Nikolay on 08.05.2026.
//

#pragma once
#include <memory>

#include "Linalg.hpp"
#include "verify/verify.hpp"

namespace nn {
class AnyLoss {
 public:
  AnyLoss() = default;
  template <class Loss>
  explicit AnyLoss(Loss loss)
      : impl_(std::make_unique<Model<std::decay_t<Loss>>>(std::move(loss))) {}
  AnyLoss(const AnyLoss& other) : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}
  AnyLoss& operator=(const AnyLoss& other) { return *this = AnyLoss(other); }

  AnyLoss(AnyLoss&& other) noexcept = default;
  AnyLoss& operator=(AnyLoss&& other) noexcept = default;
  ~AnyLoss() = default;

  Scalar operator()(const Matrix& prediction, const Matrix& target) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->operator()(prediction, target);
  }
  Matrix gradient(const Matrix& prediction, const Matrix& target) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->gradient(prediction, target);
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual std::unique_ptr<Concept> clone() const = 0;
    virtual Scalar operator()(const Matrix& prediction, const Matrix& target) const = 0;
    virtual Matrix gradient(const Matrix& prediction, const Matrix& target) const = 0;
  };
  template <class Loss>
  class Model final : public Concept {
   public:
    explicit Model(Loss loss) : loss_(std::move(loss)) {}
    std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<Loss>>(loss_); }
    Scalar operator()(const Matrix& prediction, const Matrix& target) const override {
      return loss_(prediction, target);
    }
    Matrix gradient(const Matrix& prediction, const Matrix& target) const override {
      return loss_.gradient(prediction, target);
    }

   private:
    Loss loss_;
  };
  std::unique_ptr<Concept> impl_;
};
}  // namespace nn
