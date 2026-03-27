//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once
#include "../include/tensor.hpp"
#include "../verify/verify.hpp"

namespace nn {

struct InputDim {
  int value;
};
struct OutputDim {
  int value;
};
struct LinLayerCache {
  Tensor input;
};

struct LinLayerGrads {
  Tensor grads_input;
  Tensor grads_weights;
  Tensor grads_bias;
};
class LinLayer {
public:
  LinLayer (InputDim input_dim, OutputDim output_dim);
  Tensor predict(const Tensor& input) const;

  std::pair<Tensor, LinLayerCache> forward(const Tensor& input) const;

  LinLayerGrads backward(const Tensor& grads_output, const LinLayerCache& cache) const;
  void update(const LinLayerGrads& grads, float learning_rate);

  int input_dim() const;
  int output_dim() const;

  const Tensor& weigts() const;
  const Tensor& bias() const;
private:
  int input_dim_;
  int output_dim_;

  Tensor weights_;
  Tensor bias_;
};
}  // namespace nn