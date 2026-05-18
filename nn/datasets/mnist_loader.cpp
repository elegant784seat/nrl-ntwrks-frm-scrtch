//
// Created by Loginov Nikolay on 18.05.2026.
//

#include "datasets/mnist_loader.hpp"
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include "verify/verify.hpp"

namespace nn {

namespace {
std::int32_t ReadInt32(std::ifstream& input) {
  unsigned char bytes[4];
  input.read(reinterpret_cast<char*>(bytes), 4);
  NN_VERIFY(input.good());

  return (static_cast<std::int32_t>(bytes[0]) << 24) |
         (static_cast<std::int32_t>(bytes[1]) << 16) |
         (static_cast<std::int32_t>(bytes[2]) << 8) |
         static_cast<std::int32_t>(bytes[3]);

}

std::string JoinPath(const std::string& directory, const std::string& filename) {
  if (!directory.empty() && directory.back() == '/') {
    return directory + filename;
  }
  return directory + "/" + filename;
}

}  // namespace

MnistLoader::MnistLoader(std::string directory)
    : directory_(std::move(directory)) {}

MnistLoader::Dataset MnistLoader::loadTrain() const {
  return load(JoinPath(directory_, "train-images.idx3-ubyte"),
              JoinPath(directory_, "train-labels.idx1-ubyte"));

}

MnistLoader::Dataset MnistLoader::loadTest() const {
  return load(JoinPath(directory_, "t10k-images.idx3-ubyte"),
              JoinPath(directory_, "t10k-labels.idx1-ubyte"));

}

MnistLoader::Dataset MnistLoader::load(const std::string& images_path,
                                       const std::string& labels_path) const {

  std::ifstream images_file(images_path, std::ios::binary);
  std::ifstream labels_file(labels_path, std::ios::binary);
  NN_VERIFY(images_file.is_open());
  NN_VERIFY(labels_file.is_open());

  const std::int32_t images_magic = ReadInt32(images_file);
  const std::int32_t images_count = ReadInt32(images_file);
  const std::int32_t rows = ReadInt32(images_file);
  const std::int32_t cols = ReadInt32(images_file);
  const std::int32_t labels_magic = ReadInt32(labels_file);
  const std::int32_t labels_count = ReadInt32(labels_file);

  NN_VERIFY(images_magic == 2051);
  NN_VERIFY(labels_magic == 2049);
  NN_VERIFY(images_count == labels_count);
  NN_VERIFY(images_count > 0);
  NN_VERIFY(rows == 28);
  NN_VERIFY(cols == 28);

  const Index samples = static_cast<Index>(images_count);
  const Index image_size = static_cast<Index>(rows * cols);
  Matrix input(samples, image_size);
  Matrix target = Matrix::Zero(samples, 10);
  for (Index sample = 0; sample < samples; ++sample) {
    for (Index pixel = 0; pixel < image_size; ++pixel) {
      unsigned char value = 0;
      images_file.read(reinterpret_cast<char*>(&value), 1);
      NN_VERIFY(images_file.good());
      input(sample, pixel) = static_cast<Scalar>(value) / 255.0F;

    }
    unsigned char label = 0;
    labels_file.read(reinterpret_cast<char*>(&label), 1);
    NN_VERIFY(labels_file.good());
    NN_VERIFY(label < 10);
    target(sample, static_cast<Index>(label)) = 1.0F;

  }

  return Dataset{
      .input = std::move(input),
      .target = std::move(target),
  };
}

}  // namespace nn