//
// Created by Loginov Nikolay on 12.04.2026.
//
#pragma once
#include <Eigen/Dense>

namespace nn {
using Scalar = float;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
using Index = Eigen::Index;
}  // namespace nn
