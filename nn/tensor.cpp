#include "tensor.hpp"

#include <utility>

#include "verify/location.hpp"
#include "verify/verify.hpp"
namespace nn {

Tensor::Tensor(int rows, int columns) : data_(rows, columns) {
  NN_VERIFY(rows >= 0);
  NN_VERIFY(columns >= 0);
  data_.setZero();
}
Tensor::Tensor(Matrix data) : data_(std::move(data)) {}

int Tensor::rows() const { return data_.rows(); }

int Tensor::columns() const { return data_.cols(); }

const Tensor::Matrix& Tensor::data() const { return data_; }
Tensor::Matrix& Tensor::data() { return data_; }

Tensor Tensor::transpose() const { return Tensor(data_.transpose().eval()); }

Tensor Tensor::matmul(const Tensor& A, const Tensor& B) {
  NN_VERIFY(A.columns() == B.rows());
  Tensor result(A.rows(), B.columns());
  result.data() = A.data() * B.data();
  return result;
}
Tensor Tensor::add(const Tensor& A, const Tensor& B) {
  NN_VERIFY(A.columns() == B.columns());
  NN_VERIFY(A.rows() == B.rows());
  Tensor result(A.rows(), A.columns());
  result.data() = A.data() + B.data();
  return result;
}

Tensor Tensor::add_row(const Tensor& A, const Tensor& row) {
  NN_VERIFY(A.columns() == row.columns());
  NN_VERIFY(row.rows() == 1);
  Tensor result(A.rows(), A.columns());
  result.data() = A.data().rowwise() + row.data().row(0);
  return result;
}
Tensor Tensor::sum_rows(const Tensor& a) {
  NN_VERIFY(a.rows() > 0);
  Tensor result(1, a.columns());
  result.data().row(0) = a.data().colwise().sum();
  return result;
}
}  // namespace nn

//
// Created by Loginov Nikolay on 22.02.2026.
//
