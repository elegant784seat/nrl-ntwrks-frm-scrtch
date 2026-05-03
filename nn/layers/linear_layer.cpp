#include "../layers/linear_layer.hpp"

namespace nn {

// LinLayer::Grad& operator+=(LinLayer::Grad& left, const LinLayer::Grad& right) {
//   left.weights = left.weights + right.weights;
//   left.bias = left.bias + right.bias;
//   return left;
// }
//
// LinLayer::Grad& operator-=(LinLayer::Grad& left, const LinLayer::Grad& right) {
//   left.weights = left.weights - right.weights;
//   left.bias = left.bias - right.bias;
//   return left;
// }
//
// LinLayer::Grad& operator*=(LinLayer::Grad& grad, Scalar scalar) {
//   grad.weights = grad.weights * scalar;
//   grad.bias = grad.bias * scalar;
//   return grad;
// }
//
// LinLayer::Grad& operator/=(LinLayer::Grad& grad, Scalar scalar) {
//   NN_VERIFY(scalar != 0);
//   grad.weights = grad.weights / scalar;
//   grad.bias = grad.bias / scalar;
//   return grad;
// }
//
// LinLayer::Grad operator+(LinLayer::Grad left, const LinLayer::Grad& right) {
//   left += right;
//   return left;
// }
//
// LinLayer::Grad operator-(LinLayer::Grad left, const LinLayer::Grad& right) {
//   left -= right;
//   return left;
// }
//
// LinLayer::Grad operator*(LinLayer::Grad grad, Scalar scalar) {
//   grad *= scalar;
//   return grad;
// }
//
// LinLayer::Grad operator*(Scalar scalar, LinLayer::Grad grad) {
//   grad *= scalar;
//   return grad;
// }
//
// LinLayer::Grad operator/(LinLayer::Grad grad, Scalar scalar) {
//   grad /= scalar;
//   return grad;
// }

LinLayer::LinLayer(In input_dim, Out output_dim, Random& random)
    : weights_(random.normalMatrix(input_dim, output_dim, 0, 0.01)),
      bias_(RowVector::Zero(output_dim)) {
  NN_VERIFY(input_dim > 0);
  NN_VERIFY(output_dim > 0);
}

Matrix LinLayer::predict(const Matrix& input) const {
  NN_VERIFY(input.cols() == input_dim());
  return (input * weights_).rowwise() + bias_;
}

LinLayer::ForwardResult LinLayer::forward(Matrix&& input) const {
  Matrix output = predict(input);
  State state{std::move(input)};
  return ForwardResult{std::any(std::move(state)), std::move(output)};
}

LinLayer::BackwardResult LinLayer::backward(const std::any& state,
                                            const Matrix& grad_output) const {
  NN_VERIFY(state.has_value());

  const auto& layer_state = std::any_cast<const State&>(state);

  NN_VERIFY(layer_state.input.cols() == input_dim());
  NN_VERIFY(grad_output.cols() == output_dim());
  NN_VERIFY(grad_output.rows() == layer_state.input.rows());

  Grad grad{
      layer_state.input.transpose() * grad_output,
      grad_output.colwise().sum(),
  };

  Matrix grad_input = grad_output * weights_.transpose();

  return BackwardResult{std::any(std::move(grad)), std::move(grad_input)};
}

void LinLayer::update(const std::any& state, const std::any& grad, std::any& optimizer,
                      std::any& cache) {
  (void)state;
  (void)cache;

  NN_VERIFY(grad.has_value());
  NN_VERIFY(optimizer.has_value());

  const auto& layer_grad = std::any_cast<const Grad&>(grad);
  const Scalar learning_rate = std::any_cast<Scalar>(optimizer);

  NN_VERIFY(learning_rate >= 0);
  NN_VERIFY(layer_grad.weights.rows() == weights_.rows());
  NN_VERIFY(layer_grad.weights.cols() == weights_.cols());
  NN_VERIFY(layer_grad.bias.rows() == bias_.rows());
  NN_VERIFY(layer_grad.bias.cols() == bias_.cols());

  weights_ -= learning_rate * layer_grad.weights;
  bias_ -= learning_rate * layer_grad.bias;
}

Index LinLayer::input_dim() const { return weights_.rows(); }

Index LinLayer::output_dim() const { return weights_.cols(); }

const Matrix& LinLayer::weights() const { return weights_; }

const RowVector& LinLayer::bias() const { return bias_; }

}  // namespace nn
