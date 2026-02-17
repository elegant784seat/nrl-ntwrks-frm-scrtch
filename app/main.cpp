#include <Eigen/Dense>
#include <iostream>

using Tensor = Eigen::MatrixXd;

int main() {
  Tensor A(2, 3);
  A << 1,2,3,
       4,5,6;

  Tensor B = A * A.transpose();

  std::cout << B << std::endl;
}