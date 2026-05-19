//
// Created by Loginov Nikolay on 19.05.2026.
//

#pragma once

#include <filesystem>

#include "datasets/dataset.hpp"

namespace nn {
class FashionMnistLoader {
 public:
  explicit FashionMnistLoader(std::filesystem::path directory);
  Dataset loadTrain() const;
  Dataset loadTest() const;

 private:
  std::filesystem::path directory_;
};

}  // namespace nn
