//
// Created by Loginov Nikolay on 19.05.2026.
//

#pragma once

#include <filesystem>

#include "datasets/dataset.hpp"

namespace nn {

struct IdxLoaderConfig {
  bool normalize = true;
  bool one_hot_labels = true;
  Index classes_count = 10;
};

Dataset LoadIdxDataset(const std::filesystem::path& images_path,
                       const std::filesystem::path& labels_path,
                       const IdxLoaderConfig& config = {});

}  // namespace nn
