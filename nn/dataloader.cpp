//
// Created by Loginov Nikolay on 05.05.2026.
//

// пиздец у меня тут стоко static_cast что мне страшно просто
#include "dataloader.hpp"

#include "layers/random.hpp"
#include "verify/verify.hpp"

namespace nn {
DataLoader::DataLoader(Matrix input, Matrix target, Index batch_size, ShuffleMode shuffle_mode)
    : input_(std::move(input)),
      target_(std::move(target)),
      indexes_(static_cast<size_t>(input.rows())),  // блять я боюсь касты юзать но вроде норм
      batch_size_(batch_size),
      shuffle_mode_(shuffle_mode) {
  NN_VERIFY(input_.rows() == target.rows());
  NN_VERIFY(batch_size_ > 0);
  std::iota(indexes_.begin(), indexes_.end(), Index{0});
  reset();
}

bool DataLoader::hasNext() const {
  return position_ <
         static_cast<Index>(indexes_.size());  // я себя тварью чувствую когда пишу слово cast
}

Batch DataLoader::next() {
  NN_VERIFY(hasNext());

  const Index remain = static_cast<Index>(indexes_.size()) - position_;  // снова это страшное слово
  const Index cur_batch_size = std::min(batch_size_, remain);

  Matrix batch_input(cur_batch_size, input_.cols());
  Matrix batch_target(cur_batch_size, target_.cols());

  for (Index row = 0; row < cur_batch_size; ++row) {
    const Index source_row = indexes_[static_cast<size_t>(position_ + row)];
    batch_input.row(row) = input_.row(source_row);
    batch_target.row(row) = target_.row(source_row);
  }
  position_ =
      position_ + cur_batch_size;  // по поводу += я вам отвечал, но я не знаю видели ли вы. Короче
                                   // я всегда пишу a = a + b, но если прям сильно быстрее += то
                                   // буду писать +=, но на сколько я знаю разницы нет почти
  return Batch{.input = std::move(batch_input), .target = std::move(batch_target)};
}

void DataLoader::reset() {
  position_ = 0;
  if (shuffle_mode_ == ShuffleMode::EveryEpoch) {
    Random::Global().shuffle(indexes_);
  }
}

Index DataLoader::samplesCount() const { return static_cast<Index>(indexes_.size()); }

Index DataLoader::batchSize() const { return batch_size_; }

}  // namespace nn
