//
// Created by Loginov Nikolay on 18.05.2026.
//

#include "datasets/mnist_loader.hpp"

#include <utility>

#include "datasets/idx_loader.hpp"

namespace nn {

MnistLoader::MnistLoader(std::filesystem::path directory) : directory_(std::move(directory)) {}

Dataset MnistLoader::loadTrain() const {
  return LoadIdxDataset(directory_ / "train-images.idx3-ubyte",
                        directory_ / "train-labels.idx1-ubyte");
}

Dataset MnistLoader::loadTest() const {
  return LoadIdxDataset(directory_ / "t10k-images.idx3-ubyte",
                        directory_ / "t10k-labels.idx1-ubyte");
}

}  // namespace nn
