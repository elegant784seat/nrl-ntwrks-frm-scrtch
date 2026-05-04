#include "run_all_tests.hpp"

#include <iostream>
#include <string>
#include <utility>

#include "Linalg.hpp"
#include "activation/any_func.hpp"
#include "activation/relu_func.hpp"
#include "activation/sigmoid_func.hpp"
#include "activation/tanh_func.hpp"
#include "layers/any_layer.hpp"
#include "layers/linear_layer.hpp"
#include "layers/nonlinear_layer.hpp"

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

}  // namespace

int run_all_tests() {
  const Status a = CheckAnyFuncWithNonLin();
  const Status b = CheckAnyLayerCpMv();
  const Status c = CheckAnyLayerWithLinLayer();
  const Status d = CheckAnyLayerWithNonLin();

  if (IsOk(a) && IsOk(b) && IsOk(c) && IsOk(d)) {
    std::cout << "Good!" << std::endl;
    return 0;
  }

  std::cout << "Bad!" << std::endl;
  return 1;
}

}  // namespace nn
