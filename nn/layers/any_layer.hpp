//
// Created by Loginov Nikolay on 31.03.2026.
//

#pragma once
#include <any>
#include <memory>

#include "Linalg.hpp"
#include "verify/verify.hpp"
namespace nn {
class AnyLayer {
 public:
  struct ForwardResult {
    std::any state;
    Matrix output;
  };

  struct BackwardResult {
    std::any grad;
    Matrix grad_input;
  };

  AnyLayer() = default;

  template <class Layer>
  AnyLayer(Layer layer) : impl_(std::make_unique<Model<std::decay_t<Layer>>>(std::move(layer))) {}

  AnyLayer(const AnyLayer& other) : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

  AnyLayer& operator=(const AnyLayer& other) { return *this = AnyLayer(other); }
  AnyLayer& operator=(AnyLayer&& other) noexcept = default;
  AnyLayer(AnyLayer&& other) noexcept = default;
  ~AnyLayer() = default;

  Matrix predict(const Matrix& input) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->predict(input);
  }

  ForwardResult forward(Matrix&& input) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->forward(std::move(input));
  }

  BackwardResult backward(const std::any& state, const Matrix& grad_output) const {
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
    virtual Matrix predict(const Matrix& input) const = 0;

    virtual ForwardResult forward(Matrix&& input) const = 0;

    virtual BackwardResult backward(const std::any& state, const Matrix& grad_output) const = 0;

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
    Matrix predict(const Matrix& input) const override { return layer_.predict(input); }

    ForwardResult forward(Matrix&& input) const override {
      auto result = layer_.forward(std::move(input));
      return ForwardResult{std::move(result.state), std::move(result.output)};
    }

    BackwardResult backward(const std::any& state, const Matrix& grad_output) const override {
      auto result = layer_.backward(state, grad_output);
      return BackwardResult{std::move(result.grad), std::move(result.grad_input)};
    }

    void update(const std::any& state, const std::any& grad, std::any& optimizer,
                std::any& cache) override {
      layer_.update(state, grad, optimizer, cache);
    }

   private:
    Layer layer_;
  };
  std::unique_ptr<Concept> impl_;
};
}  // namespace nn
