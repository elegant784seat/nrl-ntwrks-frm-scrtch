//
// Created by Loginov Nikolay on 18.05.2026.
//

#pragma once
#include <memory>

#include "any_types.hpp"

namespace nn {
class AnyOptimizer {
public:
  AnyOptimizer() = default;

  template <class Optimizer>
  explicit AnyOptimizer(Optimizer optimizer)
      : impl_(std::make_unique<Model<std::decay_t<Optimizer>>>(std::move(optimizer))) {}

  AnyOptimizer(const AnyOptimizer& other) : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

  AnyOptimizer& operator=(const AnyOptimizer& other) { return *this = AnyOptimizer(other); }

  AnyOptimizer(AnyOptimizer&& other) noexcept = default;
  AnyOptimizer& operator=(AnyOptimizer&& other) noexcept = default;
  ~AnyOptimizer() = default;

  AnyGrad make(const AnyGrad& grad, AnyCache& cache) {
    return impl_->make(grad, cache);
  }

  AnyCache initCache(const AnyGrad& example_grad) const {
    return impl_->initCache(example_grad);
  }

  bool hasValue() const noexcept {
    return impl_ != nullptr;
  }

  explicit operator bool() const noexcept {
    return hasValue();
  }


 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual std::unique_ptr<Concept> clone() const = 0;
    virtual AnyGrad make(const AnyGrad& grad, AnyCache& cache) = 0;
    virtual AnyCache initCache(const AnyGrad& example_grad) const = 0;
  };
  template <class Optimizer>
  struct Model final : Concept {
    explicit Model(Optimizer optimizer) : optimizer_(std::move(optimizer)) {}

    std::unique_ptr<Concept> clone() const override {
      return std::make_unique<Model<Optimizer>>(optimizer_);
    }

    AnyGrad make(const AnyGrad& grad, AnyCache& cache) override {
      return optimizer_.make(grad, cache);
    }

    AnyCache initCache(const AnyGrad& example_grad) const override {
      return optimizer_.initCache(example_grad);
    }
    Optimizer optimizer_;
  };
  std::unique_ptr<Concept> impl_;
};
}  // namespace nn
