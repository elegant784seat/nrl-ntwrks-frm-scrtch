//
// Created by Loginov Nikolay on 19.05.2026.
//

#include "datasets/fashion_mnist_loader.hpp"

#include <utility>

#include "datasets/idx_loader.hpp"

namespace nn {

FashionMnistLoader::FashionMnistLoader(std::filesystem::path directory)
    : directory_(std::move(directory)) {}

Dataset FashionMnistLoader::loadTrain() const {
  return LoadIdxDataset(directory_ / "train-images-idx3-ubyte",

                        directory_ / "train-labels-idx1-ubyte");
}

Dataset FashionMnistLoader::loadTest() const {
  return LoadIdxDataset(directory_ / "t10k-images-idx3-ubyte",

                        directory_ / "t10k-labels-idx1-ubyte");
}

}  // namespace nn
