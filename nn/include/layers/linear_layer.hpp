//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once
#include <any>

#include "../include/tensor.hpp"
#include "../verify/verify.hpp"

namespace nn {

struct InputDim {
  int value;
};
struct OutputDim {
  int value;
};

struct LinLayerState {
  Tensor input;
};

struct LinLayerGrad {
  Tensor grad_weights;
  Tensor grad_bias;

  LinLayerGrad operator+(const LinLayerGrad& other) const;
  LinLayerGrad operator-(const LinLayerGrad& other) const;
  LinLayerGrad operator*(float scalar) const;
  LinLayerGrad operator/(float scalar) const;
};

class LinLayer {
 public:
  LinLayer(InputDim input_dim, OutputDim output_dim);
  Tensor predict(const Tensor& input) const;

  std::pair<std::any, Tensor> forward(const Tensor& input) const;

  std::pair<std::any, Tensor> backward(const std::any& state, const Tensor& grad_output) const;
  void update(const std::any& state, const std::any& grad, std::any& optimizer, std::any& cache);

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