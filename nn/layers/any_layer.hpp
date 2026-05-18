//
// Created by Loginov Nikolay on 31.03.2026.
//

#pragma once
#include <memory>

#include "Linalg.hpp"
#include "any_types.hpp"
#include "optimizer/any_optimizer.hpp"
#include "verify/verify.hpp"

namespace nn {
class AnyLayer {
 public:
  struct ForwardResult {
    AnyState state;
    Matrix output;
  };

  struct BackwardResult {
    AnyGrad grad;
    Matrix grad_input;
  };

  AnyLayer() = default;

  template <class Layer>
  explicit AnyLayer(Layer layer)
      : impl_(std::make_unique<Model<std::decay_t<Layer>>>(std::move(layer))) {}

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

  BackwardResult backward(const AnyState& state, const Matrix& grad_output) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->backward(state, grad_output);
  }

  void update(const AnyState& state, const AnyGrad& grad, AnyOptimizer& optimizer,
              AnyCache& cache) {
    NN_VERIFY(impl_ != nullptr);
    impl_->update(state, grad, optimizer, cache);
  }

  AnyCache initCache(const AnyOptimizer& optimizer) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->initCache(optimizer);
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual Matrix predict(const Matrix& input) const = 0;

    virtual ForwardResult forward(Matrix&& input) const = 0;

    virtual BackwardResult backward(const AnyState& state, const Matrix& grad_output) const = 0;

    virtual void update(const AnyState& state, const AnyGrad& grad, AnyOptimizer& optimizer,
                        AnyCache& cache) = 0;

    virtual std::unique_ptr<Concept> clone() const = 0;

    virtual AnyCache initCache(const AnyOptimizer& optimizer) const = 0;
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
      return ForwardResult{.state = AnyState(std::move(result.state)),
                           .output = std::move(result.output)};
    }

    BackwardResult backward(const AnyState& state, const Matrix& grad_output) const override {
      using State = typename Layer::State;

      auto result = layer_.backward(state.get<State>(), grad_output);
      return BackwardResult{.grad = AnyGrad(std::move(result.grad)),
                            .grad_input = std::move(result.grad_input)};
    }

    void update(const AnyState& state, const AnyGrad& grad, AnyOptimizer& optimizer,
                AnyCache& cache) override {
      using State = typename Layer::State;
      using Grad = typename Layer::Grad;
      using Cache = typename Layer::Cache;

      layer_.update(state.get<State>(), grad.get<Grad>(), optimizer, cache.get<Cache>());
    }

    AnyCache initCache(const AnyOptimizer& optimizer) const override {
      return AnyCache(layer_.initCache(optimizer));
    }

   private:
    Layer layer_;
  };
  std::unique_ptr<Concept> impl_;
};
}  // namespace nn
