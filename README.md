# Neural Networks from Scratch

C++20 library for building and training fully connected neural networks from scratch.

Russian project name: **Нейросети с нуля**.

---

## Overview

This project is an educational neural network library written in modern C++.

The goal is to implement the core parts of neural network training manually, without high-level machine learning frameworks such as PyTorch or TensorFlow.

The project includes:

- fully connected neural network layers;
- activation functions;
- loss functions;
- optimizers;
- mini-batch data loading;
- MNIST / Fashion-MNIST IDX loading;
- forward propagation;
- backpropagation;
- training loop;
- type-erased C++ interfaces.

---

## Features

- Fully connected neural networks
- Mini-batch training
- Forward and backward propagation
- Manual gradient computation
- MNIST support
- Fashion-MNIST support
- Type-erased interfaces
- Modern C++20 codebase
- CMake build system
- Git submodules for third-party dependencies
- Dataset download scripts

---

## Implemented Components

### Activation Functions

- ReLU
- Sigmoid
- Tanh
- `AnyFunc`

### Layers

- `LinearLayer`
- `NonLinearLayer`
- `AnyLayer`

### Loss Functions

- Mean Squared Error
- Softmax Cross-Entropy
- `AnyLoss`

### Optimizers

- SGD
- Adam
- `AnyOptimizer`

### Data Loading

- Generic `DataLoader`
- IDX dataset loader
- MNIST-compatible loader
- Batch iteration
- Optional shuffling

### Training

- Training loop
- Forward pass
- Backward pass
- Gradient propagation
- Parameter updates

---

## Dependencies

The project uses Git submodules:

- Eigen
- EigenRand

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/elegant784seat/nrl-ntwrks-frm-scrtch.git
cd nrl-ntwrks-frm-scrtch
```

If the repository was cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Check that submodules were downloaded correctly:

```bash
ls external/eigen/Eigen
ls external/eigenrand
```

If `external/eigen` or `external/eigenrand` is empty, run:

```bash
rm -rf external/eigen external/eigenrand
git submodule update --init --recursive
```

---

## Dataset Setup

Datasets are not stored in the repository.

Download MNIST:

```bash
./scripts/download_mnist.sh
```

Download Fashion-MNIST:

```bash
./scripts/download_fashion_mnist.sh
```

If scripts are not executable:

```bash
chmod +x scripts/download_mnist.sh scripts/download_fashion_mnist.sh
```

Expected MNIST structure:

```text
data/mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

Expected Fashion-MNIST structure:

```text
data/fashion_mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

Both datasets use the same IDX format, so the same MNIST-style loader can be used for both.

---

## Project Structure

```text
.
├── app/                          # Demo application and manual tests
├── data/                         # Dataset directories
│   ├── mnist/
│   └── fashion_mnist/
├── external/                     # Third-party libraries as Git submodules
│   ├── eigen/
│   └── eigenrand/
├── nn/
│   ├── activation/               # Activation functions
│   ├── datasets/                 # IDX and MNIST-style dataset loaders
│   ├── layers/                   # Neural network layers
│   ├── loss/                     # Loss functions
│   ├── optimizer/                # Optimizers
│   ├── train/                    # Training loop
│   ├── verify/                   # Runtime checks
│   ├── any_types.hpp
│   ├── any_value.hpp
│   ├── dataloader.hpp
│   ├── except.hpp
│   └── Linalg.hpp
├── scripts/
│   ├── download_mnist.sh
│   └── download_fashion_mnist.sh
├── CMakeLists.txt
└── README.md
```

---

## Build

### Release

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

### Debug

```bash
cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
```

---

## Run

### Run all tests

```bash
cmake --build cmake-build-release --target run_all_tests
./cmake-build-release/run_all_tests
```

### Run demo application

```bash
cmake --build cmake-build-release --target nrl_ntwrks_frm_scrtch
./cmake-build-release/nrl_ntwrks_frm_scrtch
```

---

## Quick Start

Full setup from a clean clone:

```bash
git clone --recurse-submodules https://github.com/elegant784seat/nrl-ntwrks-frm-scrtch.git
cd nrl-ntwrks-frm-scrtch

git submodule sync --recursive
git submodule update --init --recursive

chmod +x scripts/download_mnist.sh scripts/download_fashion_mnist.sh
./scripts/download_mnist.sh
./scripts/download_fashion_mnist.sh

cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release

./cmake-build-release/run_all_tests
```

---

## Example

```cpp
#include "activation/relu_func.hpp"
#include "layers/linear_layer.hpp"
#include "layers/nonlinear_layer.hpp"
#include "loss/softmax_cross_entropy_loss.hpp"
#include "optimizer/sgd.hpp"

using namespace nn;

int main() {
  LinearLayer layer(In{784}, Out{128});
  ReluFunc relu;
  SoftmaxCrossEntropyLoss loss;
  Sgd optimizer(0.01f);

  return 0;
}
```

---

## Mathematical Background

For one linear layer, the forward pass is:

```math
Y = XW + b
```

where:

- `X` is the input matrix;
- `W` is the weight matrix;
- `b` is the bias vector;
- `Y` is the output matrix.

A non-linear activation function is applied element-wise:

```math
A = f(Y)
```

---

## Mean Squared Error

```math
L(y_pred, y) =
1 / n * sum((y_pred_i - y_i)^2)
```

Gradient:

```math
dL / dy_pred =
2 / n * (y_pred - y)
```

---

## Softmax

```math
softmax(z_i) =
exp(z_i) / sum(exp(z_j))
```

Numerically stable form:

```math
softmax(z_i) =
exp(z_i - max(z)) / sum(exp(z_j - max(z)))
```

---

## Cross-Entropy Loss

```math
L(y_pred, y) =
- sum(y_i * log(y_pred_i))
```

For Softmax Cross-Entropy:

```math
dL / dz =
y_pred - y
```

---

## Backpropagation

Backpropagation is based on the chain rule:

```math
dL / dx =
dL / dy * dy / dx
```

For the linear layer:

```math
Y = XW + b
```

Gradients:

```math
dL / dW =
X^T * dL / dY
```

```math
dL / db =
sum(dL / dY)
```

```math
dL / dX =
dL / dY * W^T
```

---

## Optimization

### SGD

```math
theta_next =
theta - alpha * grad
```

### Adam

```math
m_t =
beta_1 * m_prev + (1 - beta_1) * g_t
```

```math
v_t =
beta_2 * v_prev + (1 - beta_2) * g_t^2
```

```math
m_hat =
m_t / (1 - beta_1^t)
```

```math
v_hat =
v_t / (1 - beta_2^t)
```

```math
theta_next =
theta - alpha * m_hat / (sqrt(v_hat) + eps)
```

---

## Architecture

The project uses:

- value semantics;
- RAII;
- type erasure;
- explicit ownership;
- modular abstractions;
- compile-time and run-time polymorphism.

Main type-erased abstractions:

- `AnyFunc`
- `AnyLayer`
- `AnyLoss`
- `AnyOptimizer`

Type erasure is used to provide value-like polymorphic interfaces without exposing inheritance-heavy APIs to the user.

---

## Training Data Flow

```text
input batch
    |
    v
forward pass
    |
    v
prediction
    |
    v
loss computation
    |
    v
loss gradient
    |
    v
backward pass
    |
    v
parameter gradients
    |
    v
optimizer update
```

---

## Troubleshooting

### Eigen/Dense file not found

Check that Eigen submodule is downloaded:

```bash
ls external/eigen/Eigen
```

If the directory is empty:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

If it still does not work:

```bash
rm -rf external/eigen external/eigenrand
git submodule update --init --recursive
```

Then rebuild from scratch:

```bash
rm -rf cmake-build-release
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

### Dataset files are missing

Run:

```bash
./scripts/download_mnist.sh
./scripts/download_fashion_mnist.sh
```

Then check:

```bash
ls data/mnist
ls data/fashion_mnist
```

### Scripts do not run

Run:

```bash
chmod +x scripts/download_mnist.sh scripts/download_fashion_mnist.sh
```

---

## Goals

- Understand neural networks from first principles
- Learn modern C++ architecture
- Build reusable ML abstractions
- Implement neural network training manually
- Work with MNIST-style datasets
- Create a clean educational codebase

---

## Future Plans

- CNN layers
- Serialization
- GPU support
- Python bindings
- Additional optimizers
- More datasets
- Parallel training
- Extended test infrastructure

---

## References

- Eigen: https://eigen.tuxfamily.org
- EigenRand: https://github.com/bab2min/EigenRand
- Sean Parent — Runtime Polymorphism: https://www.youtube.com/watch?v=QGcVXgEVMJg
- cppreference: https://en.cppreference.com
- Dima Trushin: https://github.com/DimaTrushin/CppCode

---

## Author

Nikolay Loginov  
HSE University — Software Engineering