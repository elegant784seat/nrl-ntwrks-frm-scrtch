# Neural Networks from Scratch

C++ library for building and training fully connected neural networks from scratch.

Russian project name: **Нейросети с нуля**.

---

## Overview

This project is an educational neural network library written in modern C++.

The main goal is to implement the core parts of neural network training manually, without using high-level machine learning frameworks such as PyTorch or TensorFlow.

The project focuses on:

- fully connected neural networks;
- forward propagation;
- backpropagation;
- loss functions;
- optimizers;
- mini-batch training;
- dataset loading;
- type erasure in C++;
- value semantics;
- clean modular API design.

The library is intended primarily for studying how neural networks work internally.

---

## Features

- Fully connected neural network components
- Mini-batch data loading
- Forward and backward propagation
- Manual gradient computation
- MNIST and Fashion-MNIST IDX dataset support
- Type-erased interfaces for activations, layers, losses and optimizers
- Modern C++20 codebase
- CMake build system
- Git submodules for third-party dependencies
- Console demo and manual tests

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
- MNIST-compatible dataset loader
- Batch iteration
- Optional shuffling

### Training

- Training loop
- Forward pass
- Backward pass
- Gradient propagation
- Parameter update step

---

## Tech Stack

- C++20
- CMake
- Eigen
- EigenRand
- Git submodules
- clang-format
- clang-tidy

---

## Dependencies

The project uses third-party dependencies through Git submodules:

- Eigen
- EigenRand

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/elegant784seat/nrl-ntwrks-frm-scrtch.git
cd nrl-ntwrks-frm-scrtch
```

If the repository was cloned without submodules, initialize them manually:

```bash
git submodule update --init --recursive
```

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
│   ├── verify/                   # Runtime verification utilities
│   ├── any_types.hpp             # Type-erased training-related aliases/types
│   ├── any_value.hpp             # Generic type-erased value holder
│   ├── dataloader.hpp            # Mini-batch DataLoader
│   ├── except.hpp                # Project exception utilities
│   └── Linalg.hpp                # Linear algebra aliases
├── scripts/
│   ├── download_mnist.sh
│   └── download_fashion_mnist.sh
├── CMakeLists.txt
└── README.md
```

---

## Dataset Setup

Datasets are not stored in the repository.

The repository contains scripts for downloading MNIST and Fashion-MNIST in IDX format.

### MNIST

Download MNIST:

```bash
./scripts/download_mnist.sh
```

Expected structure:

```text
data/mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

### Fashion-MNIST

Download Fashion-MNIST:

```bash
./scripts/download_fashion_mnist.sh
```

Expected structure:

```text
data/fashion_mnist/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

Both datasets use the same IDX file structure, so the same MNIST-style loader can be used for both.

---

## Build

### Release Build

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

### Debug Build

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

int main() {
  nn::LinearLayer layer(784, 128);
  nn::ReluFunc relu;
  nn::SoftmaxCrossEntropyLoss loss;
  nn::Sgd optimizer(0.01f);

  return 0;
}
```

---

## Mathematical Background

The library implements a standard fully connected neural network.

For one linear layer, the forward pass is:

```math
Y = XW + b
```

where:

- `X` is the input matrix;
- `W` is the weight matrix;
- `b` is the bias vector;
- `Y` is the output matrix.

A non-linear activation function is then applied element-wise:

```math
A = f(Y)
```

---

## Mean Squared Error

The Mean Squared Error loss is defined as:

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

Softmax converts raw model outputs into probabilities:

```math
softmax(z_i) =
exp(z_i) / sum(exp(z_j))
```

For numerical stability, the implementation can use the shifted form:

```math
softmax(z_i) =
exp(z_i - max(z)) / sum(exp(z_j - max(z)))
```

---

## Cross-Entropy Loss

Cross-Entropy loss is defined as:

```math
L(y_pred, y) =
- sum(y_i * log(y_pred_i))
```

For Softmax Cross-Entropy, the gradient with respect to logits simplifies to:

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

the gradients are:

### Gradient with respect to weights

```math
dL / dW =
X^T * dL / dY
```

### Gradient with respect to bias

```math
dL / db =
sum(dL / dY)
```

### Gradient with respect to input

```math
dL / dX =
dL / dY * W^T
```

---

## Optimization

### Stochastic Gradient Descent

The SGD update rule:

```math
theta_next =
theta - alpha * grad
```

where:

- `theta` is a trainable parameter;
- `alpha` is the learning rate;
- `grad` is the gradient of the loss function.

---

## Adam Optimizer

First moment estimate:

```math
m_t =
beta_1 * m_prev + (1 - beta_1) * g_t
```

Second moment estimate:

```math
v_t =
beta_2 * v_prev + (1 - beta_2) * g_t^2
```

Bias correction:

```math
m_hat =
m_t / (1 - beta_1^t)
```

```math
v_hat =
v_t / (1 - beta_2^t)
```

Final parameter update:

```math
theta_next =
theta - alpha * m_hat / (sqrt(v_hat) + eps)
```

Adam adapts the update scale for each parameter independently and is commonly used for training neural networks.

---

## Architecture

The project uses modern C++ design techniques:

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

This makes it possible to store different implementations behind one uniform interface while preserving copy and move semantics.

---

## Design Principles

The library follows several core design principles:

- neural network components should be reusable;
- training state should be separated from layer parameters;
- forward propagation should explicitly return state required for backward propagation;
- gradients should be represented separately from trainable parameters;
- optimizers should be independent from layers;
- interfaces should remain explicit and predictable;
- implementation details should remain hidden behind abstractions.

---

## Training Data Flow

Typical training pipeline:

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