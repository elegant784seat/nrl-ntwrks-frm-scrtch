#include "../include/activation/relu_func.hpp"
namespace nn {

Tensor ReluFunc::operator()(const Tensor& input) const {
  Tensor output(input.rows(), input.columns());
  output.data() = input.data().cwiseMax(0);
  return output;
}

Tensor ReluFunc::derivative(const Tensor& input, const Tensor& output) const {
  (void)output;
  Tensor result(input.rows(), input.columns());
  result.data() = (input.data().array() > 0).template cast<float>().matrix();
  // я не знаю хороший ли это каст, но вроде ниче такого,
  // просто вместо того чтоб в цикле делать статик каст
  // я использую эйдженовский каст
  return result;
}
}  // namespace nn
// Created by Loginov Nikolay on 31.03.2026.
//
