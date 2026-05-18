//
// Created by Loginov Nikolay on 18.05.2026.
//

#include "train_loop.hpp"

#include <chrono>
#include <iostream>

#include "verify/verify.hpp"

namespace nn {
namespace {
Index ArgMaxRow(const Matrix& matrix, Index row) {
  NN_VERIFY(row >= 0);
  NN_VERIFY(row < matrix.rows());
  NN_VERIFY(matrix.cols() > 0);

  Index best_col = 0;
  Scalar best_value = matrix(row, 0);

  for (Index col = 1; col < matrix.cols(); ++col) {
    if (matrix(row, col) > best_value) {
      best_value = matrix(row, col);
      best_col = col;
    }
  }
  return best_col;
}
}  // namespace

TrainLoop::TrainLoop(TrainConfig config) : train_config_(config) {
  NN_VERIFY(train_config_.epochs > 0);
}

std::vector<EpochStat> TrainLoop::fit(Network& network, DataLoader& train_loader, AnyLoss& loss,
                                      AnyOptimizer& optimizer) const {
  std::vector<EpochStat> stats;
  stats.reserve(train_config_.epochs);

  Network::Cache cache = network.initCache(optimizer);

  for (Index epoch = 1; epoch <= train_config_.epochs; ++epoch) {
    EpochStat stat = trainEpoch(network, train_loader, loss, optimizer, cache, epoch);
    stats.push_back(stat);

    if (train_config_.log_mode == LogMode::Epoch) {
      printStat(stat);
    }
  }
  return stats;
}

EpochStat TrainLoop::trainEpoch(Network& network, DataLoader& train_loader, AnyLoss& loss,
                                AnyOptimizer& optimizer, Network::Cache& cache, Index epoch) const {
  const auto start = std::chrono::steady_clock::now();
  train_loader.reset();

  Scalar total_loss = 0;
  Scalar total_accuracy = 0;
  Index batch_count = 0;

  while (train_loader.hasNext()) {
    Batch batch = train_loader.next();
    auto forward_result = network.forward(std::move(batch.input));

    const Scalar batch_loss = loss(forward_result.output, batch.target);
    Matrix grad_output = loss.gradient(forward_result.output, batch.target);

    auto backward_result = network.backward(forward_result.state, grad_output);

    network.update(forward_result.state, backward_result.grad, optimizer, cache);
    total_loss += batch_loss;

    if (train_config_.metrics_mode == MetricsMode::LossAndAccuracy) {
      total_accuracy += accuracy(forward_result.output, batch.target);
    }
    ++batch_count;
  }
  NN_VERIFY(batch_count > 0);
  const auto finish = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(finish - start).count();

  EpochStat stat{
      .epoch = epoch,
      .loss = total_loss / static_cast<Scalar>(batch_count),
      .accuracy = 0,
      .batch_count = batch_count,
      .seconds = seconds,
  };

  if (train_config_.metrics_mode == MetricsMode::LossAndAccuracy) {
    stat.accuracy = total_accuracy / static_cast<Scalar>(batch_count);
  }

  return stat;
}

EvalStat TrainLoop::evaluate(Network& network, DataLoader& loader, AnyLoss& loss) const {
  loader.reset();

  Scalar total_loss = 0;
  Scalar total_accuracy = 0;
  Index batch_count = 0;

  while (loader.hasNext()) {
    Batch batch = loader.next();
    Matrix prediction = network.predict(batch.input);
    total_loss += loss(prediction, batch.target);

    if (train_config_.metrics_mode == MetricsMode::LossAndAccuracy) {
      total_accuracy += accuracy(prediction, batch.target);
    }
    ++batch_count;
  }

  NN_VERIFY(batch_count > 0);

  EvalStat stat{
      .loss = total_loss / static_cast<Scalar>(batch_count),
      .accuracy = 0,
      .batch_count = batch_count,
  };

  if (train_config_.metrics_mode == MetricsMode::LossAndAccuracy) {
    stat.accuracy = total_accuracy / static_cast<Scalar>(batch_count);
  }

  return stat;
}

Scalar TrainLoop::accuracy(const Matrix& prediction, const Matrix& target) const {
  NN_VERIFY(prediction.rows() == target.rows());
  NN_VERIFY(prediction.cols() == target.cols());
  NN_VERIFY(prediction.rows() > 0);
  NN_VERIFY(prediction.cols() > 0);

  Index correct = 0;
  for (Index row = 0; row < prediction.rows(); ++row) {
    const Index predicted_class = ArgMaxRow(prediction, row);
    const Index target_class = ArgMaxRow(target, row);

    if (predicted_class == target_class) {
      ++correct;
    }
  }

  return static_cast<Scalar>(correct) / static_cast<Scalar>(prediction.rows());
}

void TrainLoop::printStat(const EpochStat& stat) const {
  std::cout << "epoch = " << stat.epoch << ", loss = " << stat.loss
            << ", accuracy = " << stat.accuracy << ", batches = " << stat.batch_count
            << ", time = " << stat.seconds << " seconds" << std::endl;
}

}  // namespace nn
