#include "run_all_tests.hpp"

#include <iostream>
#include <string>
#include <utility>

#include "Linalg.hpp"
#include "activation/any_func.hpp"
#include "activation/relu_func.hpp"
#include "activation/sigmoid_func.hpp"
#include "activation/tanh_func.hpp"
#include "dataloader.hpp"
#include "layers/any_layer.hpp"
#include "layers/linear_layer.hpp"
#include "layers/nonlinear_layer.hpp"
#include "loss/any_loss.hpp"
#include "loss/mse_loss.hpp"
#include "loss/softmax_cross_entropy_loss.hpp"
#include "optimizer/sgd.hpp"
#include "optimizer/true_adam.hpp"

namespace nn {
namespace {

void PrintHead(const std::string& str) {
  std::cout << "///////////////////" << std::endl;
  std::cout << str << std::endl;
  std::cout << "///////////////////" << std::endl;
}

void PrintNamedMatrix(const std::string& name, const Matrix& matrix) {
  std::cout << name << ":" << std::endl;
  std::cout << matrix << std::endl << std::endl;
}

enum class Status {
  Ok,
  Error,
};

bool IsOk(Status status) { return status == Status::Ok; }

Status CheckAnyFuncWithNonLin() {
  PrintHead("Check AnyFunc and NonLinLayer");

  std::cout << "Relu" << std::endl;
  {
    AnyFunc func = ReluFunc{};
    NonLinLayer layer(std::move(func));

    Matrix x(1, 3);
    x << -1, 0, 2;

    PrintNamedMatrix("predict(x)", layer.predict(x));

    auto [state, y_forward] = layer.forward(std::move(x));
    PrintNamedMatrix("forward(x)", y_forward);

    Matrix grad_out(1, 3);
    grad_out << 1, 1, 1;

    auto [grad_param, grad_input] = layer.backward(state, grad_out);
    PrintNamedMatrix("grad_input", grad_input);
  }

  std::cout << "Sigmoid" << std::endl;
  {
    AnyFunc func = SigmoidFunc{};
    NonLinLayer layer(std::move(func));

    Matrix x(1, 3);
    x << -1, 0, 1;

    PrintNamedMatrix("predict(x)", layer.predict(x));

    auto [state, y_forward] = layer.forward(std::move(x));
    PrintNamedMatrix("forward(x)", y_forward);

    Matrix grad_out(1, 3);
    grad_out << 1, 1, 1;

    auto [grad_param, grad_input] = layer.backward(state, grad_out);
    PrintNamedMatrix("grad_input", grad_input);
  }

  std::cout << "Tanh" << std::endl;
  {
    AnyFunc func = TanhFunc{};
    NonLinLayer layer(std::move(func));

    Matrix x(1, 3);
    x << -1, 0, 1;

    PrintNamedMatrix("predict(x)", layer.predict(x));

    auto [state, y_forward] = layer.forward(std::move(x));
    PrintNamedMatrix("forward(x)", y_forward);

    Matrix grad_out(1, 3);
    grad_out << 1, 1, 1;

    auto [grad_param, grad_input] = layer.backward(state, grad_out);
    PrintNamedMatrix("grad_input", grad_input);
  }

  return Status::Ok;
}

Status CheckAnyLayerWithNonLin() {
  PrintHead("Check AnyLayer and NonLinLayer");

  AnyFunc func = ReluFunc{};
  NonLinLayer nonlin(std::move(func));
  AnyLayer layer(std::move(nonlin));

  Matrix x(1, 3);
  x << -2, 0, 3;

  PrintNamedMatrix("predict(x)", layer.predict(x));

  auto [state, y_forward] = layer.forward(std::move(x));
  PrintNamedMatrix("forward(x)", y_forward);

  Matrix grad_out(1, 3);
  grad_out << 1, 1, 1;

  auto [grad_param, grad_input] = layer.backward(state, grad_out);
  PrintNamedMatrix("grad_input", grad_input);

  return Status::Ok;
}

Status CheckAnyLayerWithLinLayer() {
  PrintHead("Check AnyLayer and LinLayer");

  LinLayer lin(In{3}, Out{2});
  AnyLayer layer(std::move(lin));

  Matrix x(1, 3);
  x << 1, 2, 3;

  PrintNamedMatrix("predict(x)", layer.predict(x));

  auto [state, y_forward] = layer.forward(std::move(x));
  PrintNamedMatrix("forward(x)", y_forward);

  Matrix grad_out(1, 2);
  grad_out << 1, 1;

  auto [grad_param, grad_input] = layer.backward(state, grad_out);
  PrintNamedMatrix("grad_input", grad_input);

  return Status::Ok;
}

Status CheckAnyLayerCpMv() {
  PrintHead("Check AnyLayer cp/mv");

  Matrix x(1, 3);
  x << -1, 0, 1;

  AnyLayer l1(NonLinLayer(AnyFunc(SigmoidFunc{})));
  AnyLayer l2 = l1;
  AnyLayer l3 = std::move(l2);

  PrintNamedMatrix("l1.predict(x)", l1.predict(x));
  PrintNamedMatrix("l3.predict(x)", l3.predict(x));

  AnyLayer l4(NonLinLayer(AnyFunc(TanhFunc{})));
  l4 = l1;
  PrintNamedMatrix("l4.predict(x)", l4.predict(x));

  AnyLayer l5(NonLinLayer(AnyFunc(ReluFunc{})));
  l5 = std::move(l4);
  PrintNamedMatrix("l5.predict(x)", l5.predict(x));

  return Status::Ok;
}

Status CheckDataLoader() {
  PrintHead("Check DataLoader");

  Matrix input(5, 2);
  input << 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F;

  Matrix target(5, 1);
  target << 0.0F, 1.0F, 0.0F, 1.0F, 0.0F;

  DataLoader loader(input, target, 2, ShuffleMode::None);

  std::cout << "samples = " << loader.samplesCount() << std::endl;
  std::cout << "batch size = " << loader.batchSize() << std::endl;

  Index batch_count = 0;
  while (loader.hasNext()) {
    Batch batch = loader.next();

    PrintNamedMatrix("batch.input", batch.input);
    PrintNamedMatrix("batch.target", batch.target);

    ++batch_count;
  }

  NN_VERIFY(batch_count == 3);

  loader.reset();
  NN_VERIFY(loader.hasNext());

  Batch first_batch = loader.next();
  NN_VERIFY(first_batch.input.rows() == 2);
  NN_VERIFY(first_batch.input.cols() == 2);
  NN_VERIFY(first_batch.target.rows() == 2);
  NN_VERIFY(first_batch.target.cols() == 1);

  return Status::Ok;
}

Status CheckAnyLoss() {
  PrintHead("Check AnyLoss");

  std::cout << "MSELoss" << std::endl;
  {
    Matrix prediction(2, 2);
    prediction << 1.0F, 2.0F, 3.0F, 4.0F;

    Matrix target(2, 2);
    target << 1.0F, 1.0F, 2.0F, 2.0F;

    AnyLoss loss(MseLoss{});

    Scalar value = loss(prediction, target);
    Matrix grad = loss.gradient(prediction, target);

    std::cout << "loss = " << value << std::endl;
    PrintNamedMatrix("gradient", grad);
  }

  std::cout << "SoftMaxCrossEntropyLoss" << std::endl;
  {
    Matrix logits(2, 3);
    logits << 2.0F, 1.0F, 0.1F, 0.5F, 2.5F, 0.3F;

    Matrix target(2, 3);
    target << 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F;

    AnyLoss loss(SoftMaxCrossEntropyLoss{});

    Scalar value = loss(logits, target);
    Matrix grad = loss.gradient(logits, target);

    std::cout << "loss = " << value << std::endl;
    PrintNamedMatrix("gradient", grad);
  }

  return Status::Ok;
}

Status CheckSgdOptimizer() {
  PrintHead("Check SgdOptimizer");

  LinLayer layer(In{2}, Out{1});

  LinLayer::Grad grad{
    .weights = Matrix::Ones(2, 1),
    .bias = RowVector::Ones(1),
};

  AnyOptimizer optimizer(Sgd{0.1F});
  AnyCache cache = optimizer.initCache(AnyGrad(grad));

  Matrix before_weights = layer.weights();
  RowVector before_bias = layer.bias();

  LinLayer::State state{
    .input = Matrix::Zero(1, 2),
};

  layer.update(state, grad, optimizer, cache);

  Matrix expected_weights = before_weights - 0.1F * grad.weights;
  RowVector expected_bias = before_bias - 0.1F * grad.bias;

  PrintNamedMatrix("before_weights", before_weights);
  PrintNamedMatrix("after_weights", layer.weights());
  PrintNamedMatrix("expected_weights", expected_weights);

  NN_VERIFY((layer.weights() - expected_weights).norm() < 1e-5F);
  NN_VERIFY((layer.bias() - expected_bias).norm() < 1e-5F);

  return Status::Ok;
}

Status CheckAdamOptimizer() {
  PrintHead("Check AdamOptimizer");

  LinLayer layer(In{2}, Out{1});

  LinLayer::Grad grad{
    .weights = Matrix::Ones(2, 1),
    .bias = RowVector::Ones(1),
};

  AnyOptimizer optimizer(Adam{0.001F});
  AnyCache cache = optimizer.initCache(AnyGrad(layer.zeroGrad()));

  Matrix before_weights = layer.weights();
  RowVector before_bias = layer.bias();

  LinLayer::State state{
    .input = Matrix::Zero(1, 2),
};

  layer.update(state, grad, optimizer, cache);

  PrintNamedMatrix("before_weights", before_weights);
  PrintNamedMatrix("after_weights", layer.weights());

  NN_VERIFY((layer.weights() - before_weights).norm() > 0);
  NN_VERIFY((layer.bias() - before_bias).norm() > 0);
  auto& adam_cache = cache.get<Adam::Cache>();
  NN_VERIFY(adam_cache.t == 1);

  NN_VERIFY((adam_cache.m.weights - 0.1F * grad.weights).norm() < 1e-5F);
  NN_VERIFY((adam_cache.m.bias - 0.1F * grad.bias).norm() < 1e-5F);

  NN_VERIFY((adam_cache.v.weights - 0.001F * grad.weights.array().square().matrix()).norm() <
            1e-5F);
  NN_VERIFY((adam_cache.v.bias - 0.001F * grad.bias.array().square().matrix()).norm() < 1e-5F);

  return Status::Ok;
}

}  // namespace

int run_all_tests() {
  const Status a = CheckAnyFuncWithNonLin();
  const Status b = CheckAnyLayerCpMv();
  const Status c = CheckAnyLayerWithLinLayer();
  const Status d = CheckAnyLayerWithNonLin();
  const Status e = CheckDataLoader();
  const Status f = CheckAnyLoss();
  const Status g = CheckSgdOptimizer();
  const Status h = CheckAdamOptimizer();

  if (IsOk(a) && IsOk(b) && IsOk(c) && IsOk(d) && IsOk(e) && IsOk(f) && IsOk(g) && IsOk(h)) {
    std::cout << "Good!" << std::endl;
    return 0;
  }

  std::cout << "Bad!" << std::endl;
  return 1;
}

}  // namespace nn
