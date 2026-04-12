//
// Created by Loginov Nikolay on 28.03.2026.
//
#include "../layers/linear_layer.hpp"

#include <EigenRand/EigenRand>

#include "../verify/verify.hpp"

namespace nn {

static void InitializeWeights(Tensor& weights) {
  Eigen::Rand::P8_mt19937_64 generator(std::random_device{}());
  weights.data() = Eigen::Rand::normal<Eigen::MatrixXf>(weights.rows(), weights.columns(),
                                                        generator, 0.0f, 0.01f);
}

static void InitializeBias(Tensor& bias) { bias.data().setZero(); }

LinLayerGrad LinLayerGrad::operator+(const LinLayerGrad& other) const {
  return LinLayerGrad{Tensor::add(grad_weights, other.grad_weights),
                      Tensor::add(grad_bias, other.grad_bias)};
}

LinLayerGrad LinLayerGrad::operator-(const LinLayerGrad& other) const {
  Tensor weights = grad_weights;
  Tensor bias = grad_bias;
  weights.data() = weights.data() - other.grad_weights.data();
  bias.data() = bias.data() - other.grad_bias.data();
  return LinLayerGrad{std::move(weights), std::move(bias)};
}

LinLayerGrad LinLayerGrad::operator*(float scalar) const {
  Tensor weights = grad_weights;
  Tensor bias = grad_bias;
  weights.data() = weights.data() * scalar;
  bias.data() = bias.data() * scalar;
  return LinLayerGrad{std::move(weights), std::move(bias)};
}

LinLayerGrad LinLayerGrad::operator/(float scalar) const {
  NN_VERIFY(scalar != 0);
  Tensor weights = grad_weights;
  Tensor bias = grad_bias;
  weights.data() = weights.data() / scalar;
  bias.data() = bias.data() / scalar;
  return LinLayerGrad{std::move(weights), std::move(bias)};
}

LinLayer::LinLayer(InputDim input_dim, OutputDim output_dim)
    : input_dim_(input_dim.value),
      output_dim_(output_dim.value),
      weights_(input_dim.value, output_dim.value),
      bias_(1, output_dim.value) {
  NN_VERIFY(input_dim_ > 0 && output_dim_ > 0);
  InitializeWeights(weights_);
  InitializeBias(bias_);
}

Tensor LinLayer::predict(const Tensor& input) const {
  NN_VERIFY(input.columns() == input_dim_);
  Tensor tmp = Tensor::matmul(input, weights_);
  Tensor output = Tensor::add_row(tmp, bias_);
  return output;
}

std::pair<std::any, Tensor> LinLayer::forward(const Tensor& input) const {
  NN_VERIFY(input.columns() == input_dim_);
  Tensor tmp = Tensor::matmul(input, weights_);
  Tensor output = Tensor::add_row(tmp, bias_);
  LinLayerState state{input};
  return {std::any(std::move(state)), std::move(output)};
}

std::pair<std::any, Tensor> LinLayer::backward(const std::any& state,
                                               const Tensor& grad_output) const {
  NN_VERIFY(state.has_value());
  const auto& layer_state = std::any_cast<const LinLayerState&>(state);
  NN_VERIFY(layer_state.input.columns() == input_dim_);
  NN_VERIFY(grad_output.columns() == output_dim_);
  NN_VERIFY(grad_output.rows() == layer_state.input.rows());

  Tensor grad_weights = Tensor::matmul(layer_state.input.transpose(), grad_output);
  Tensor grad_bias = Tensor::sum_rows(grad_output);
  Tensor grad_input = Tensor::matmul(grad_output, weights_.transpose());

  LinLayerGrad grad{std::move(grad_weights), std::move(grad_bias)};

  return {std::any(std::move(grad)), std::move(grad_input)};
}

void LinLayer::update(const std::any& state, const std::any& grad, std::any& optimizer,
                      std::any& cache) {
  (void)state, (void)cache;

  NN_VERIFY(grad.has_value());
  NN_VERIFY(optimizer.has_value());

  const auto& layer_grad = std::any_cast<const LinLayerGrad&>(grad);
  const float learning_rate = std::any_cast<float>(optimizer);
  NN_VERIFY(learning_rate >= 0);

  NN_VERIFY(layer_grad.grad_weights.rows() == weights_.rows());
  NN_VERIFY(layer_grad.grad_weights.columns() == weights_.columns());
  NN_VERIFY(layer_grad.grad_bias.rows() == bias_.rows());
  NN_VERIFY(layer_grad.grad_bias.columns() == bias_.columns());

  weights_.data() = weights_.data() - learning_rate * layer_grad.grad_weights.data();
  bias_.data() = bias_.data() - learning_rate * layer_grad.grad_bias.data();
}

int LinLayer::input_dim() const { return input_dim_; }

int LinLayer::output_dim() const { return output_dim_; }

const Tensor& LinLayer::weigts() const { return weights_; }

const Tensor& LinLayer::bias() const { return bias_; }

}  // namespace nn
