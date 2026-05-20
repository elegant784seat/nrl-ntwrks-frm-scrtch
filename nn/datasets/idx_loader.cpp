#include "datasets/idx_loader.hpp"

#include <cstdint>
#include <fstream>

#include "verify/verify.hpp"

namespace nn {
namespace {

std::uint32_t ToByte(char byte) {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(byte));
}

std::uint32_t ReadBigEndianUInt32(std::ifstream& input) {
  char bytes[4];
  input.read(bytes, 4);
  NN_VERIFY(input.good());

  return (ToByte(bytes[0]) << 24) | (ToByte(bytes[1]) << 16) | (ToByte(bytes[2]) << 8) |
         ToByte(bytes[3]);
}

std::ifstream OpenBinaryFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  NN_VERIFY(input.is_open());
  return input;
}

unsigned char ReadByte(std::ifstream& input) {
  char byte = 0;
  input.read(&byte, 1);
  NN_VERIFY(input.good());

  return static_cast<unsigned char>(byte);
}

}  // namespace

Dataset LoadIdxDataset(const std::filesystem::path& images_path,
                       const std::filesystem::path& labels_path, const IdxLoaderConfig& config) {
  std::ifstream images = OpenBinaryFile(images_path);
  std::ifstream labels = OpenBinaryFile(labels_path);

  const std::uint32_t images_magic = ReadBigEndianUInt32(images);
  const std::uint32_t images_count = ReadBigEndianUInt32(images);
  const std::uint32_t rows = ReadBigEndianUInt32(images);
  const std::uint32_t cols = ReadBigEndianUInt32(images);

  const std::uint32_t labels_magic = ReadBigEndianUInt32(labels);
  const std::uint32_t labels_count = ReadBigEndianUInt32(labels);

  NN_VERIFY(images_magic == 2051);
  NN_VERIFY(labels_magic == 2049);
  NN_VERIFY(images_count == labels_count);
  NN_VERIFY(rows > 0);
  NN_VERIFY(cols > 0);
  NN_VERIFY(config.classes_count > 0);

  const Index samples_count = static_cast<Index>(images_count);
  const Index image_size = static_cast<Index>(rows * cols);

  Dataset dataset;
  dataset.input = Matrix::Zero(samples_count, image_size);

  if (config.one_hot_labels) {
    dataset.target = Matrix::Zero(samples_count, config.classes_count);
  } else {
    dataset.target = Matrix::Zero(samples_count, 1);
  }

  for (Index sample = 0; sample < samples_count; ++sample) {
    for (Index pixel_index = 0; pixel_index < image_size; ++pixel_index) {
      const unsigned char pixel = ReadByte(images);

      Scalar value = static_cast<Scalar>(pixel);
      if (config.normalize) {
        value /= static_cast<Scalar>(255);
      }

      dataset.input(sample, pixel_index) = value;
    }

    const unsigned char label = ReadByte(labels);
    const Index label_index = static_cast<Index>(label);

    NN_VERIFY(label_index < config.classes_count);

    if (config.one_hot_labels) {
      dataset.target(sample, label_index) = static_cast<Scalar>(1);
    } else {
      dataset.target(sample, 0) = static_cast<Scalar>(label_index);
    }
  }

  return dataset;
}

}  // namespace nn
