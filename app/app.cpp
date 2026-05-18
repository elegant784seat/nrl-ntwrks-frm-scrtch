//
// Created by Loginov Nikolay on 18.05.2026.
//

#include "app.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "activation/any_func.hpp"
#include "activation/relu_func.hpp"
#include "activation/sigmoid_func.hpp"
#include "activation/tanh_func.hpp"
#include "dataloader.hpp"
#include "datasets/mnist_loader.hpp"
#include "layers/any_layer.hpp"
#include "layers/linear_layer.hpp"
#include "layers/network.hpp"
#include "layers/nonlinear_layer.hpp"
#include "loss/any_loss.hpp"
#include "loss/mse_loss.hpp"
#include "loss/softmax_cross_entropy_loss.hpp"
#include "optimizer/any_optimizer.hpp"
#include "optimizer/sgd.hpp"
#include "optimizer/true_adam.hpp"
#include "train/train_loop.hpp"

namespace nn::app {
namespace {

enum class ActivationChoice {
  Relu = 1,
  Sigmoid = 2,
  Tanh = 3,
};

void ClearInput() {
  std::cin.clear();
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

Index ReadIndex(const std::string& prompt, Index min_value, Index max_value) {
  while (true) {
    std::cout << prompt;

    Index value = 0;
    if (std::cin >> value && value >= min_value && value <= max_value) {
      return value;
    }

    std::cout << "Invalid value. Expected integer in [" << min_value << ", "
              << max_value << "]." << std::endl;
    ClearInput();
  }
}

Scalar ReadScalar(const std::string& prompt, Scalar min_value, Scalar max_value) {
  while (true) {
    std::cout << prompt;

    Scalar value = 0;
    if (std::cin >> value && value >= min_value && value <= max_value) {
      return value;
    }

    std::cout << "Invalid value. Expected number in [" << min_value << ", "
              << max_value << "]." << std::endl;
    ClearInput();
  }
}

ActivationChoice ReadActivation(Index layer_number) {
  std::cout << "\nActivation for hidden layer " << layer_number << ":\n";
  std::cout << "  1. ReLU\n";
  std::cout << "  2. Sigmoid\n";
  std::cout << "  3. Tanh\n";

  return static_cast<ActivationChoice>(
      ReadIndex("Choose activation: ", 1, 3));
}

AnyLayer MakeActivationLayer(ActivationChoice choice) {
  switch (choice) {
    case ActivationChoice::Relu:
      return AnyLayer(NonLinLayer(AnyFunc(ReluFunc{})));

    case ActivationChoice::Sigmoid:
      return AnyLayer(NonLinLayer(AnyFunc(SigmoidFunc{})));

    case ActivationChoice::Tanh:
      return AnyLayer(NonLinLayer(AnyFunc(TanhFunc{})));
  }

  NN_VERIFY(false);
}

Network BuildDefaultNetwork(Index input_dim, Index output_dim) {
  return Network({
      AnyLayer(LinLayer(In{input_dim}, Out{128})),
      AnyLayer(NonLinLayer(AnyFunc(ReluFunc{}))),
      AnyLayer(LinLayer(In{128}, Out{output_dim})),
  });
}

Network BuildCustomNetwork(Index input_dim, Index output_dim) {
  std::cout << "\nCustom fully-connected network builder\n";
  std::cout << "Input dim: " << input_dim << '\n';
  std::cout << "Output dim: " << output_dim << '\n';

  const Index hidden_count = ReadIndex("Hidden layers count [1..5]: ", 1, 5);

  std::vector<AnyLayer> layers;
  layers.reserve(static_cast<std::size_t>(2 * hidden_count + 1));

  Index current_dim = input_dim;

  for (Index layer = 1; layer <= hidden_count; ++layer) {
    const Index hidden_dim = ReadIndex("Hidden layer size [1..2048]: ", 1, 2048);
    const ActivationChoice activation = ReadActivation(layer);

    layers.push_back(AnyLayer(LinLayer(In{current_dim}, Out{hidden_dim})));
    layers.push_back(MakeActivationLayer(activation));

    current_dim = hidden_dim;
  }

  layers.push_back(AnyLayer(LinLayer(In{current_dim}, Out{output_dim})));
  return Network(std::move(layers));
}

Network BuildNetworkFromMenu(Index input_dim, Index output_dim) {
  std::cout << "\nNetwork architecture:\n";
  std::cout << "  1. Default network\n";
  std::cout << "  2. Custom network\n";

  const Index choice = ReadIndex("Choose architecture: ", 1, 2);

  if (choice == 1) {
    return BuildDefaultNetwork(input_dim, output_dim);
  }

  return BuildCustomNetwork(input_dim, output_dim);
}

AnyLoss BuildLossFromMenu() {
  std::cout << "\nLoss function:\n";
  std::cout << "  1. SoftmaxCrossEntropy\n";
  std::cout << "  2. MSE\n";

  const Index choice = ReadIndex("Choose loss: ", 1, 2);

  if (choice == 1) {
    return AnyLoss(SoftMaxCrossEntropyLoss{});
  }

  return AnyLoss(MseLoss{});
}

AnyOptimizer BuildOptimizerFromMenu() {
  std::cout << "\nOptimizer:\n";
  std::cout << "  1. SGD\n";
  std::cout << "  2. Adam\n";

  const Index choice = ReadIndex("Choose optimizer: ", 1, 2);
  const Scalar learning_rate =
      ReadScalar("Learning rate [0.000001..0.2]: ", 0.000001F, 0.2F);

  if (choice == 1) {
    return AnyOptimizer(Sgd(learning_rate));
  }

  return AnyOptimizer(Adam(learning_rate));
}

TrainConfig BuildTrainConfigFromMenu() {
  std::cout << "\nTraining configuration:\n";

  return TrainConfig{
      .epochs = ReadIndex("Epochs [1..100]: ", 1, 100),
      .log_mode = LogMode::Epoch,
      .metrics_mode = MetricsMode::LossAndAccuracy,
  };
}

Index ReadBatchSize() {
  return ReadIndex("Batch size [1..512]: ", 1, 512);
}

}  // namespace

int RunMnistDemo() {
  std::cout << "Neural Networks from Scratch\n";
  std::cout << "Interactive MNIST training application\n";

  MnistLoader mnist("data/mnist");

  std::cout << "Loading MNIST...\n";
  MnistLoader::Dataset train = mnist.loadTrain();
  MnistLoader::Dataset test = mnist.loadTest();

  const Index input_dim = train.input.cols();
  const Index output_dim = train.target.cols();

  const Index batch_size = ReadBatchSize();

  DataLoader train_loader(train.input, train.target, batch_size,
                          ShuffleMode::EveryEpoch);

  DataLoader test_loader(test.input, test.target, batch_size, ShuffleMode::None);

  Network network = BuildNetworkFromMenu(input_dim, output_dim);
  AnyLoss loss = BuildLossFromMenu();
  AnyOptimizer optimizer = BuildOptimizerFromMenu();
  TrainLoop loop(BuildTrainConfigFromMenu());

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