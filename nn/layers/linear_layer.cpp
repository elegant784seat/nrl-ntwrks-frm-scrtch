#include "../layers/linear_layer.hpp"

namespace nn {

LinLayer::Grad& operator+=(LinLayer::Grad& left, const LinLayer::Grad& right) {
  left.weights += right.weights;
  left.bias += right.bias;
  return left;
}

LinLayer::Grad& operator-=(LinLayer::Grad& left, const LinLayer::Grad& right) {
  left.weights -= right.weights;
  left.bias -= right.bias;
  return left;
}

LinLayer::Grad& operator*=(LinLayer::Grad& grad, Scalar scalar) {
  grad.weights *= scalar;
  grad.bias *= scalar;
  return grad;
}

LinLayer::Grad& operator/=(LinLayer::Grad& grad, Scalar scalar) {
  NN_VERIFY(scalar != 0);
  grad.weights /= scalar;
  grad.bias /= scalar;
  return grad;
}

LinLayer::Grad operator+(LinLayer::Grad left, const LinLayer::Grad& right) {
  left += right;
  return left;
}

LinLayer::Grad operator-(LinLayer::Grad left, const LinLayer::Grad& right) {
  left -= right;
  return left;
}

LinLayer::Grad operator*(LinLayer::Grad grad, Scalar scalar) {
  grad *= scalar;
  return grad;
}

LinLayer::Grad operator*(Scalar scalar, LinLayer::Grad grad) {
  grad *= scalar;
  return grad;
}

LinLayer::Grad operator/(LinLayer::Grad grad, Scalar scalar) {
  grad /= scalar;
  return grad;
}

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

  return ForwardResult{.state = State{.input = std::move(input)}, .output = std::move(output)};
}

LinLayer::BackwardResult LinLayer::backward(const State& state, const Matrix& grad_output) const {
  NN_VERIFY(state.input.cols() == input_dim());
  NN_VERIFY(grad_output.cols() == output_dim());
  NN_VERIFY(grad_output.rows() == state.input.rows());

  Grad grad{
      .weights = state.input.transpose() * grad_output,
      .bias = grad_output.colwise().sum(),
  };

  Matrix grad_input = grad_output * weights_.transpose();

  return BackwardResult{.grad = std::move(grad), .grad_input = std::move(grad_input)};
}

void LinLayer::update(const State& state, const Grad& grad, AnyOptimizer& optimizer, Cache& cache) {
  (void)state;

  NN_VERIFY(optimizer.hasValue());

  NN_VERIFY(grad.weights.rows() == weights_.rows());
  NN_VERIFY(grad.weights.cols() == weights_.cols());
  NN_VERIFY(grad.bias.rows() == bias_.rows());
  NN_VERIFY(grad.bias.cols() == bias_.cols());

  const Grad d_grad = optimizer.make(AnyGrad(grad), cache).get<Grad>();

  NN_VERIFY(d_grad.weights.rows() == weights_.rows());
  NN_VERIFY(d_grad.weights.cols() == weights_.cols());
  NN_VERIFY(d_grad.bias.rows() == bias_.rows());
  NN_VERIFY(d_grad.bias.cols() == bias_.cols());

  weights_ -= d_grad.weights;
  bias_ -= d_grad.bias;
}

Index LinLayer::input_dim() const { return weights_.rows(); }

Index LinLayer::output_dim() const { return weights_.cols(); }

const Matrix& LinLayer::weights() const { return weights_; }

const RowVector& LinLayer::bias() const { return bias_; }

LinLayer::Grad LinLayer::zeroGrad() const {
  return Grad{
      .weights = Matrix::Zero(weights_.rows(), weights_.cols()),
      .bias = RowVector::Zero(bias_.cols()),
  };
}

}  // namespace nn
