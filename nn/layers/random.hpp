//
// Created by Loginov Nikolay on 20.04.2026.
//
#pragma once

#include <EigenRand/EigenRand>
#include <cstdint>

#include "Linalg.hpp"

namespace nn {

class Random {
  using Generator = Eigen::Rand::P8_mt19937_64;

 public:
  explicit Random(std::uint64_t seed = 42) : generator_(seed) {}
  static Random& Global();
  Matrix normalMatrix(Index rows, Index cols, Scalar mean, Scalar stddev);
  RowVector normalRowVector(Index size, Scalar mean, Scalar stddev);
  void shuffle(std::vector<Index>& indices);

 private:
  Generator generator_;
};

}  // namespace nn
