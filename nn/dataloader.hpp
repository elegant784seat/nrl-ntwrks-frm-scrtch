//
// Created by Loginov Nikolay on 04.05.2026.
//

#pragma once
#include <vector>

#include "Linalg.hpp"

namespace nn {
enum class ShuffleMode {
  None,
  EveryEpoch,
};

struct Batch {
  Matrix input;
  Matrix target;
};

class DataLoader {
 public:
  DataLoader(Matrix input, Matrix target, Index batch_size,
             ShuffleMode shuffle_mode = ShuffleMode::EveryEpoch);

  bool hasNext() const;
  Batch next();
  void reset();

  Index samplesCount() const;
  Index batchSize() const;

 private:
  Matrix input_;
  Matrix target_;
  std::vector<Index> indexes_;
  Index batch_size_;
  Index position_ = 0;
  ShuffleMode shuffle_mode_;
};

}  // namespace nn
