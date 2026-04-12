//
// Created by Loginov Nikolay on 31.03.2026.
//

#pragma once
#include <any>
#include <memory>

#include "tensor.hpp"
#include "verify/verify.hpp"
namespace nn {
class AnyLayer {
 public:
  AnyLayer() = delete;

  template <class Layer>
  AnyLayer(Layer layer) : impl_(std::make_unique<Model<std::decay_t<Layer>>>(std::move(layer))) {}

  AnyLayer(const AnyLayer& other) : impl_(other.impl_->clone()) { NN_VERIFY(impl_ != nullptr); }

  AnyLayer& operator=(const AnyLayer& other) {
    if (this == &other) return *this;
    AnyLayer tmp(other);
    swap(tmp);
    return *this;
  }
  AnyLayer& operator=(AnyLayer&& other) noexcept = default;
  AnyLayer(AnyLayer&& other) noexcept = default;
  ~AnyLayer() = default;
  void swap(AnyLayer& other) noexcept {
    std::swap(impl_, other.impl_);
  }

  Tensor predict(const Tensor& input) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->predict(input);
  }

  std::pair<std::any, Tensor> forward(const Tensor& input) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->forward(input);
  }

  std::pair<std::any, Tensor> backward(const std::any& state, const Tensor& grad_output) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->backward(state, grad_output);
  }

  void update(const std::any& state, const std::any& grad, std::any& optimizer, std::any& cache) {
    NN_VERIFY(impl_ != nullptr);
    impl_->update(state, grad, optimizer, cache);
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual Tensor predict(const Tensor& input) const = 0;

    virtual std::pair<std::any, Tensor> forward(const Tensor& input) const = 0;

    virtual std::pair<std::any, Tensor> backward(const std::any& state,
                                                 const Tensor& grad_output) const = 0;

    virtual void update(const std::any& state, const std::any& grad, std::any& optimizer,
                        std::any& cache) = 0;

    virtual std::unique_ptr<Concept> clone() const = 0;
  };
  template <class Layer>
  class Model final : public Concept {
   public:
    explicit Model(const Layer& layer) : layer_(layer) {}
    explicit Model(Layer&& layer) : layer_(std::move(layer)) {}

    std::unique_ptr<Concept> clone() const override {
      return std::make_unique<Model<Layer>>(layer_);
    }
    Tensor predict(const Tensor& input) const override { return layer_.predict(input); }

    std::pair<std::any, Tensor> forward(const Tensor& input) const override {
      return layer_.forward(input);
    }

    std::pair<std::any, Tensor> backward(const std::any& state,
                                         const Tensor& grad_output) const override {
      return layer_.backward(state, grad_output);
    }

    void update(const std::any& state, const std::any& grad, std::any& optimizer,
                std::any& cache) override {
      layer_.update(state, grad, optimizer, cache);
    }

   private:
    Layer layer_;
  };

 private:
  std::unique_ptr<Concept> impl_;
};
inline void swap(AnyLayer& left, AnyLayer& right) noexcept {
  left.swap(right);
}
}  // namespace nn
