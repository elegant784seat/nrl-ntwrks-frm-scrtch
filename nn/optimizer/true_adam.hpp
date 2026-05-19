//
// Created by Loginov Nikolay on 18.05.2026.
//

#pragma once
#include "layers/linear_layer.hpp"

namespace nn {
class Adam {
 public:
  struct Cache {
    LinLayer::Grad m;
    LinLayer::Grad v;
    Index t = 0;
  };

  explicit Adam(Scalar learning_rate = 0.001, Scalar beta1 = 0.9, Scalar beta2 = 0.999,
                Scalar eps = 1e-8)
      : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(eps) {
    NN_VERIFY(learning_rate > 0);
    NN_VERIFY(beta1 >= 0 && beta1 < 1);
    NN_VERIFY(beta2 >= 0 && beta2 < 1);
    NN_VERIFY(eps > 0);
  }

  AnyGrad make(const AnyGrad& grad, AnyCache& cache) const {
    const auto& g = grad.get<LinLayer::Grad>();
    auto& adam_cache = cache.get<Cache>();

    NN_VERIFY(g.weights.rows() == adam_cache.m.weights.rows());
    NN_VERIFY(g.weights.cols() == adam_cache.m.weights.cols());
    NN_VERIFY(g.bias.cols() == adam_cache.m.bias.cols());

    ++adam_cache.t;

    adam_cache.m.weights =
        beta1_ * adam_cache.m.weights + (1 - beta1_) * g.weights;

    adam_cache.m.bias =
        beta1_ * adam_cache.m.bias + (1 - beta1_) * g.bias;

    adam_cache.v.weights =
        beta2_ * adam_cache.v.weights +
        (1 - beta2_) * g.weights.array().square().matrix();

    adam_cache.v.bias =
        beta2_ * adam_cache.v.bias +
        (1 - beta2_) * g.bias.array().square().matrix();

    const Scalar bias_correction1 =
        1 - static_cast<Scalar>(std::pow(beta1_, adam_cache.t));

    const Scalar bias_correction2 =
        1 - static_cast<Scalar>(std::pow(beta2_, adam_cache.t));

    LinLayer::Grad m_hat = adam_cache.m / bias_correction1;
    LinLayer::Grad v_hat = adam_cache.v / bias_correction2;
    LinLayer::Grad d_grad{
        .weights =
            learning_rate_ *
            (m_hat.weights.array() /
             (v_hat.weights.array().sqrt() + eps_))
                .matrix(),
        .bias =
            learning_rate_ *
            (m_hat.bias.array() /
             (v_hat.bias.array().sqrt() + eps_))
                .matrix(),
    };
    return AnyGrad(std::move(d_grad));
  }

  AnyCache initCache(const AnyGrad& grad) const {
    const auto& g = grad.get<LinLayer::Grad>();
    return AnyCache(Cache{
        .m =
            LinLayer::Grad{
                .weights = Matrix::Zero(g.weights.rows(), g.weights.cols()),
                .bias = RowVector::Zero(g.bias.cols()),
            },
        .v =
            LinLayer::Grad{
                .weights = Matrix::Zero(g.weights.rows(), g.weights.cols()),
                .bias = RowVector::Zero(g.bias.cols()),
            },
        .t = 0,
    });
  }

 private:
  Scalar learning_rate_;
  Scalar beta1_;
  Scalar beta2_;
  Scalar eps_;
};
}  // namespace nn
