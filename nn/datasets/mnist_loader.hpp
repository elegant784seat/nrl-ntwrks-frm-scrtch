//
// Created by Loginov Nikolay on 18.05.2026.
//

#pragma once
#include <string>

#include "Linalg.hpp"

namespace nn {
class MnistLoader {
 public:
  struct Dataset {
    Matrix input;
    Matrix target;
  };

  explicit MnistLoader(std::string directory);
  Dataset loadTrain() const;
  Dataset loadTest() const;

 private:
  Dataset load(const std::string& images_path, const std::string& labels_path) const;
  std::string directory_;
};
}  // namespace nn
