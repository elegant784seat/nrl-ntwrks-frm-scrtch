#include <iostream>

#include "activation/any_func.hpp"
#include "activation/relu_func.hpp"
#include "activation/sigmoid_func.hpp"
#include "activation/tanh_func.hpp"
#include "layers/any_layer.hpp"
#include "layers/linear_layer.hpp"
#include "layers/nonlinear_layer.hpp"
#include "verify/verify.hpp"
#include "Linalg.hpp"
#include "tensor.hpp"


namespace {
void PrintTensor(const std::string& name, const nn::Tensor& tensor) {
  std::cout << name << ":" << std::endl << tensor.data() << std::endl << std::endl;
  ;
}

void PrintHead(const std::string& str) {
  std::cout << "///////////////////" << std::endl;
  std::cout << str << std::endl;
  std::cout << "///////////////////" << std::endl;
}

enum class Status {
  Norm,
  Strem,
};

bool Ok(Status status) { return status == Status::Norm; }

Status CheckAnyFuncWithNonLin() {
  PrintHead("Check AnyFunc and NonLinLayer");
  {
    std::cout << "Relu" << std::endl;

    nn::AnyFunc func = nn::ReluFunc{};
    nn::NonLinLayer layer(std::move(func));

    nn::Tensor x(1, 3);
    x.data()(0, 0) = -1;
    x.data()(0, 1) = 0;
    x.data()(0, 2) = 2;

    nn::Tensor y_predict = layer.predict(x);
    PrintTensor("predict(x)", y_predict);

    auto[st, y_forward] = layer.forward(x);
    PrintTensor("forward(x)", y_forward);

    nn::Tensor grad_out(1, 3);
    grad_out.data()(0, 0) = 1;
    grad_out.data()(0, 1) = 1;
    grad_out.data()(0, 2) = 1;

    auto [grad_param, grad_input] = layer.backward(st, grad_out);
    PrintTensor("grad_input", grad_input);
  }
  {
    std::cout << "Sigmoid" << std::endl;

    nn::AnyFunc func = nn::SigmoidFunc{};
    nn::NonLinLayer layer(std::move(func));

    nn::Tensor x(1, 3);
    x.data()(0, 0) = -1;
    x.data()(0, 1) = 0;
    x.data()(0, 2) = 1;

    nn::Tensor y_predict = layer.predict(x);
    PrintTensor("predict(x)", y_predict);

    auto[st, y_forward] = layer.forward(x);
    PrintTensor("forward(x)", y_forward);

    nn::Tensor grad_out(1, 3);
    grad_out.data()(0, 0) = 1;
    grad_out.data()(0, 1) = 1;
    grad_out.data()(0, 2) = 1;

    auto [grad_param, grad_input] = layer.backward(st, grad_out);
    PrintTensor("grad_input", grad_input);
  }
  {
    std::cout << "Tanh" << std::endl;

    nn::AnyFunc func = nn::TanhFunc{};
    nn::NonLinLayer layer(std::move(func));

    nn::Tensor x(1, 3);
    x.data()(0, 0) = -1;
    x.data()(0, 1) = 0;
    x.data()(0, 2) = 1;

    nn::Tensor y_predict = layer.predict(x);
    PrintTensor("predict(x)", y_predict);

    auto[st, y_forward] = layer.forward(x);
    PrintTensor("forward(x)", y_forward);

    nn::Tensor grad_out(1, 3);
    grad_out.data()(0, 0) = 1;
    grad_out.data()(0, 1) = 1;
    grad_out.data()(0, 2) = 1;

    auto [grad_param, grad_input] = layer.backward(st, grad_out);
    PrintTensor("grad_input", grad_input);
  }
  return Status::Norm;
}

Status CheckAnyLayerWithNonLin() {
  PrintHead("Check AnyLayer and NonLinerLayer");

  nn::AnyFunc func = nn::ReluFunc{};
  nn::NonLinLayer nonlin(std::move(func));
  nn::AnyLayer layer(std::move(nonlin));

  nn::Tensor x(1, 3);
  x.data()(0, 0) = -2;
  x.data()(0, 1) = 0;
  x.data()(0, 2) = 3;

  nn::Tensor y_predict = layer.predict(x);
  PrintTensor("predict(x)", y_predict);

  auto [state, y_forward] = layer.forward(x);
  PrintTensor("forward(x)", y_forward);

  nn::Tensor grad_out(1, 3);
  grad_out.data()(0, 0) = 1;
  grad_out.data()(0, 1) = 1;
  grad_out.data()(0, 2) = 1;

  auto [grad_param, grad_input] = layer.backward(state, grad_out);
  PrintTensor("grad_input", grad_input);

  return Status::Norm;
}

Status CheckAnyLayerWithLinLayer() {
  PrintHead("Check AnyLayer and LinLayer");

  nn::LinLayer lin(nn::InputDim{3}, nn::OutputDim{2});
  nn::AnyLayer layer(std::move(lin));

  nn::Tensor x(1, 3);
  x.data()(0, 0) = 1;
  x.data()(0, 1) = 2;
  x.data()(0, 2) = 3;

  nn::Tensor y_predict = layer.predict(x);
  PrintTensor("predict(x)", y_predict);

  auto [state, y_forward] = layer.forward(x);
  PrintTensor("forward(x)", y_forward);

  nn::Tensor grad_out(1, 2);
  grad_out.data()(0, 0) = 1;
  grad_out.data()(0, 1) = 1;

  auto [grad_param, grad_input] = layer.backward(state, grad_out);
  PrintTensor("grad_input", grad_input);
  return Status::Norm;
}
Status CheckAnyLayerCpMv() {
  PrintHead("Check AnyLayer cp/mv");

  nn::Tensor x(1, 3);
  x.data()(0, 0) = -1;
  x.data()(0, 1) = 0;
  x.data()(0, 2) = 1;

  nn::AnyLayer l1(nn::NonLinLayer(nn::AnyFunc(nn::SigmoidFunc{})));
  nn::AnyLayer l2 = l1;
  nn::AnyLayer l3 = std::move(l2);

  PrintTensor("l1.predict(x)", l1.predict(x));
  PrintTensor("l3.predict(x)", l3.predict(x));

  nn::AnyLayer l4(nn::NonLinLayer(nn::AnyFunc(nn::TanhFunc{})));
  l4 = l1;
  PrintTensor("l4.predict(x)", l4.predict(x));

  nn::AnyLayer l5(nn::NonLinLayer(nn::AnyFunc(nn::ReluFunc{})));
  l5 = std::move(l4);
  PrintTensor("l5.predict(x)", l5.predict(x));

  return Status::Norm;
}
}

int main() {
  Status a = CheckAnyFuncWithNonLin();
  Status b = CheckAnyLayerCpMv();
  Status c = CheckAnyLayerWithLinLayer();
  Status d = CheckAnyLayerWithNonLin();
  if (Ok(a) and Ok(b) and Ok(c) and Ok(d)) {
    std::cout << "Good!" << std::endl;
    return 0;
  }
  std::cout << "Bad!" << std::endl;
  return 1;
}
