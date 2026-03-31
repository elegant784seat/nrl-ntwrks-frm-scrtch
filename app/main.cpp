#include <any>
#include <iostream>

#include "activation/relu_func.hpp"
#include "activation/sigmoid_func.hpp"
#include "activation/tanh_func.hpp"
#include "any_func.hpp"
#include "layers/nonlinear_layer.hpp"
#include "tensor.hpp"

void PrintTensor(const std::string& name, const nn::Tensor& tensor) {
  std::cout << name << ":\n" << tensor.data() << "\n\n";
}

void CheckRelu() {
  std::cout << "ReLU" << std::endl;
  ;

  nn::AnyFunc func = nn::ReluFunc{};
  nn::NonLinLayer layer(std::move(func));

  nn::Tensor x(1, 3);
  x.data()(0, 0) = -1.0f;
  x.data()(0, 1) = 0.0f;
  x.data()(0, 2) = 2.0f;

  nn::Tensor y_predict = layer.predict(x);
  PrintTensor("predict(x)", y_predict);

  auto [state, y_forward] = layer.forward(x);
  PrintTensor("forward(x)", y_forward);

  nn::Tensor grad_output(1, 3);
  grad_output.data()(0, 0) = 1.0f;
  grad_output.data()(0, 1) = 1.0f;
  grad_output.data()(0, 2) = 1.0f;

  auto [grad_params, grad_input] = layer.backward(state, grad_output);
  PrintTensor("grad_input", grad_input);
}

void CheckSigmoid() {
  std::cout << "Sigmoid" << std::endl;
  ;

  nn::AnyFunc func = nn::SigmoidFunc{};
  nn::NonLinLayer layer(std::move(func));

  nn::Tensor x(1, 3);
  x.data()(0, 0) = -1.0f;
  x.data()(0, 1) = 0.0f;
  x.data()(0, 2) = 1.0f;

  nn::Tensor y_predict = layer.predict(x);
  PrintTensor("predict(x)", y_predict);

  auto [state, y_forward] = layer.forward(x);
  PrintTensor("forward(x)", y_forward);

  nn::Tensor grad_output(1, 3);
  grad_output.data()(0, 0) = 1.0f;
  grad_output.data()(0, 1) = 1.0f;
  grad_output.data()(0, 2) = 1.0f;

  auto [grad_params, grad_input] = layer.backward(state, grad_output);
  PrintTensor("grad_input", grad_input);
}

void CheckTanh() {
  std::cout << "Tanh" << std::endl;

  nn::AnyFunc func = nn::TanhFunc{};
  nn::NonLinLayer layer(std::move(func));

  nn::Tensor x(1, 3);
  x.data()(0, 0) = -1.0f;
  x.data()(0, 1) = 0.0f;
  x.data()(0, 2) = 1.0f;

  nn::Tensor y_predict = layer.predict(x);
  PrintTensor("predict(x)", y_predict);

  auto [state, y_forward] = layer.forward(x);
  PrintTensor("forward(x)", y_forward);

  nn::Tensor grad_output(1, 3);
  grad_output.data()(0, 0) = 1.0f;
  grad_output.data()(0, 1) = 1.0f;
  grad_output.data()(0, 2) = 1.0f;

  auto [grad_params, grad_input] = layer.backward(state, grad_output);
  PrintTensor("grad_input", grad_input);
}

int main() {
  CheckRelu();
  CheckSigmoid();
  CheckTanh();
  return 0;
}