//
// Created by Loginov Nikolay on 18.05.2026.
//

#include "datasets/mnist_loader.hpp"

#include <filesystem>
#include <iostream>
#include <utility>

#include "datasets/idx_loader.hpp"

namespace nn {

MnistLoader::MnistLoader(std::filesystem::path directory) : directory_(std::move(directory)) {}

Dataset MnistLoader::loadTrain() const {
  const std::filesystem::path images_path = directory_ / "train-images-idx3-ubyte";
  const std::filesystem::path labels_path = directory_ / "train-labels-idx1-ubyte";

  std::cout << "Opening train images: " << std::filesystem::absolute(images_path) << std::endl;
  std::cout << "Opening train labels: " << std::filesystem::absolute(labels_path) << std::endl;

  return LoadIdxDataset(images_path, labels_path);
}

Dataset MnistLoader::loadTest() const {
  const std::filesystem::path images_path = directory_ / "t10k-images-idx3-ubyte";
  const std::filesystem::path labels_path = directory_ / "t10k-labels-idx1-ubyte";

  std::cout << "Opening test images: " << std::filesystem::absolute(images_path) << std::endl;
  std::cout << "Opening test labels: " << std::filesystem::absolute(labels_path) << std::endl;

  return LoadIdxDataset(images_path, labels_path);
}

}  // namespace nn
