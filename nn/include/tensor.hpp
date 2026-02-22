#pragma once
#include <Eigen/Dense>
#include <string>

namespace nn {
class Tensor {
public:
  using Matrix = Eigen::MatrixXf;
  Tensor() = default;
  Tensor(int rows, int columns);
  explicit Tensor(Matrix data);

  int rows() const;
  int columns() const;

  const Matrix& data() const;
  Matrix& data();

  Tensor transpose() const;

  Tensor matrixmultiplication(const Tensor& A, const Tensor& B);
  Tensor matrixadd(const Tensor& A, const Tensor& B);
  Tensor add_row(const Tensor& A, const Tensor& row);
  Tensor sum_rows(const Tensor& a);

private:
  Matrix data_;
};
}