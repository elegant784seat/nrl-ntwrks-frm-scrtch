//
// Created by Loginov Nikolay on 18.05.2026.
//


#include "app.hpp"

#include <iostream>

#include "activation/any_func.hpp"
#include "activation/relu_func.hpp"
#include "dataloader.hpp"
#include "datasets/mnist_loader.hpp"
#include "layers/any_layer.hpp"
#include "layers/linear_layer.hpp"
#include "layers/network.hpp"
#include "layers/nonlinear_layer.hpp"
#include "loss/any_loss.hpp"
#include "loss/softmax_cross_entropy_loss.hpp"
#include "optimizer/true_adam.hpp"
#include "optimizer/any_optimizer.hpp"
#include "optimizer/true_adam.hpp"
#include "train/train_loop.hpp"

namespace nn::app {

int RunMnistDemo() {
  std::cout << "Neural Networks from Scratch: MNIST demo" << std::endl;

  MnistLoader mnist("../data/mnist");

  std::cout << "Loading MNIST..." << std::endl;
  MnistLoader::Dataset train = mnist.loadTrain();
  MnistLoader::Dataset test = mnist.loadTest();

  DataLoader train_loader(train.input, train.target, 64, ShuffleMode::EveryEpoch);
  DataLoader test_loader(test.input, test.target, 64, ShuffleMode::None);

  Network network({
      AnyLayer(LinLayer(In{784}, Out{128})),
      AnyLayer(NonLinLayer(AnyFunc(ReluFunc{}))),
      AnyLayer(LinLayer(In{128}, Out{10})),
  });

  AnyLoss loss(SoftMaxCrossEntropyLoss{});
  AnyOptimizer optimizer(Adam(0.001F));

  TrainLoop loop(TrainConfig{
      .epochs = 5,
      .log_mode = LogMode::Epoch,
      .metrics_mode = MetricsMode::LossAndAccuracy,
  });

  EvalStat before = loop.evaluate(network, test_loader, loss);
  std::cout << "Before: test loss = " << before.loss
            << ", test accuracy = " << before.accuracy << std::endl;

  loop.fit(network, train_loader, loss, optimizer);
  EvalStat after = loop.evaluate(network, test_loader, loss);
  std::cout << "After: test loss = " << after.loss
            << ", test accuracy = " << after.accuracy << std::endl;

  return 0;
}

}  // namespace nn::app