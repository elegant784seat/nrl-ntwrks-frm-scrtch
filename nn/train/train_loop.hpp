//
// Created by Loginov Nikolay on 18.05.2026.
//

#pragma once
#include <vector>

#include "Linalg.hpp"
#include "dataloader.hpp"
#include "layers/network.hpp"
#include "loss/any_loss.hpp"

namespace nn {
enum class LogMode {
  None,
  Epoch,
};

enum class MetricsMode {
  LossOnly,
  LossAndAccuracy,
};

struct TrainConfig {
  Index epochs = 1;
  LogMode log_mode = LogMode::Epoch;
  MetricsMode metrics_mode = MetricsMode::LossAndAccuracy;
};

struct EpochStat {
  Index epoch = 0;
  Scalar loss = 0;
  Scalar accuracy = 0;
  Index batch_count = 0;
  double seconds = 0;
};

struct EvalStat {
  Scalar loss = 0;
  Scalar accuracy = 0;
  Index batch_count = 0;
};

class TrainLoop {
 public:
  explicit TrainLoop(TrainConfig config = {});

  std::vector<EpochStat> fit(Network& network, DataLoader& train_loader, AnyLoss& loss,
                             AnyOptimizer& optimizer) const;

  EvalStat evaluate(Network& network, DataLoader& loader, AnyLoss& loss) const;

 private:
  EpochStat trainEpoch(Network& network, DataLoader& train_loader, AnyLoss& loss,
                       AnyOptimizer& optimizer, Network::Cache& cache, Index epoch) const;

  Scalar accuracy(const Matrix& prediction, const Matrix& target) const;

  void printStat(const EpochStat& stat) const;

  TrainConfig train_config_;
};
}  // namespace nn
