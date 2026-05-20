# Neural Networks from Scratch

C++ library for building and training fully connected neural networks from scratch.

Project name in Russian: **Нейросети с нуля**.

---

## Overview

This project is an educational neural network library written in modern C++.

The main goal of the project is to implement neural network components manually without using high-level machine learning frameworks such as PyTorch or TensorFlow.

The library is designed for studying:

- forward propagation;
- backpropagation;
- gradient descent;
- optimizers;
- neural network architecture;
- modern C++ abstractions;
- type erasure;
- value semantics;
- modular API design.

The project focuses on understanding how neural networks work internally instead of relying on existing ML ecosystems.

---

## Features

- Fully connected neural networks
- Mini-batch training
- Forward and backward propagation
- Modular neural network architecture
- MNIST-style dataset support
- Type-erased interfaces
- Modern C++20 codebase
- CMake build system
- Git submodules for dependencies
- Educational focus on neural network internals

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

### Data Loading

- Generic `DataLoader`
- MNIST dataset loader
- Batch iteration
- Optional shuffling

### Training

- Forward pass
- Backward pass
- Gradient propagation
- Parameter updates
- Training loop

---

## Tech Stack

- C++20
- CMake
- Eigen
- EigenRand
- clang-format
- clang-tidy
- Git submodules

---

## Dependencies

The project uses third-party libraries through Git submodules:

- Eigen
- EigenRand

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/elegant784seat/nrl-ntwrks-frm-scrtch.git
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## Project Structure

```text
.
├── app/                          # Demo application and manual tests
├── data/                         # Dataset files
├── external/                     # Third-party libraries
│   ├── eigen/
│   └── eigenrand/
├── nn/
│   ├── activation/               # Activation functions
│   ├── datasets/                 # Dataset loaders
│   ├── layers/                   # Neural network layers
│   ├── loss/                     # Loss functions
│   ├── optimizer/                # Optimizers
│   ├── train/                    # Training loop
│   └── Linalg.hpp                # Linear algebra aliases
├── CMakeLists.txt
└── README.md
```

---

## Build

### Clone Repository

```bash
git clone --recurse-submodules https://github.com/elegant784seat/nrl-ntwrks-frm-scrtch.git
cd nrl-ntwrks-frm-scrtch
```

If submodules were not initialized:

```bash
git submodule update --init --recursive
```

---

### Release Build

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
```

---

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

---

### Run demo application

```bash
cmake --build cmake-build-release --target nrl_ntwrks_frm_scrtch
./cmake-build-release/nrl_ntwrks_frm_scrtch
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
L(y_{pred}, y) =
\frac{1}{n}
\sum_{i=1}^{n}
(y_{pred,i} - y_i)^2
```

Gradient:

```math
\frac{\partial L}{\partial y_{pred}} =
\frac{2}{n}(y_{pred} - y)
```

---

## Softmax

Softmax converts raw model outputs into probabilities:

```math
softmax(z_i) =
\frac{e^{z_i}}
{\sum_{j=1}^{k} e^{z_j}}
```

For numerical stability, the implementation may use the shifted form:

```math
softmax(z_i) =
\frac{
e^{z_i - \max(z)}
}{
\sum_{j=1}^{k} e^{z_j - \max(z)}
}
```

---

## Cross-Entropy Loss

Cross-Entropy loss is defined as:

```math
L(y_{pred}, y) =
-
\sum_{i=1}^{k}
y_i \log(y_{pred,i})
```

For Softmax Cross-Entropy, the gradient simplifies to:

```math
\frac{\partial L}{\partial z} =
y_{pred} - y
```

---

## Backpropagation

Backpropagation is based on the chain rule:

```math
\frac{\partial L}{\partial x} =
\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial x}
```

For the linear layer:

```math
Y = XW + b
```

the gradients are:

### Gradient with respect to weights

```math
\frac{\partial L}{\partial W} =
X^T
\frac{\partial L}{\partial Y}
```

---

### Gradient with respect to bias

```math
\frac{\partial L}{\partial b} =
\sum_{i=1}^{n}
\frac{\partial L}{\partial Y_i}
```

---

### Gradient with respect to input

```math
\frac{\partial L}{\partial X} =
\frac{\partial L}{\partial Y}
W^T
```

---

## Optimization

### Stochastic Gradient Descent

The SGD update rule:

```math
\theta_{t+1} =
\theta_t
-
\alpha
\nabla_{\theta} L
```

where:

- `theta` is a trainable parameter;
- `alpha` is the learning rate;
- `∇L` is the gradient of the loss function.

---

## Adam Optimizer

First moment estimate:

```math
m_t =
\beta_1 m_{t-1}
+
(1 - \beta_1) g_t
```

Second moment estimate:

```math
v_t =
\beta_2 v_{t-1}
+
(1 - \beta_2) g_t^2
```

Bias correction:

```math
\hat{m}_t =
\frac{m_t}{1 - \beta_1^t}
```

```math
\hat{v}_t =
\frac{v_t}{1 - \beta_2^t}
```

Final parameter update:

```math
\theta_t =
\theta_{t-1}
-
\alpha
\frac{
\hat{m}_t
}{
\sqrt{\hat{v}_t} + \varepsilon
}
```

Adam adapts learning rates for each parameter independently and usually converges faster than plain SGD.

---

## Architecture

The project actively uses modern C++ design techniques:

- value semantics;
- RAII;
- type erasure;
- explicit ownership;
- modular abstractions;
- compile-time and run-time polymorphism.

Main abstractions:

- `AnyFunc`
- `AnyLayer`
- `AnyLoss`
- `AnyOptimizer`

The project uses type erasure to provide value-like polymorphic interfaces without exposing inheritance-heavy APIs to the user.

This architecture allows different implementations to be stored and composed uniformly while preserving copy/move semantics and modularity.

---

## Design Principles

The library follows several core principles:

- neural network components should be reusable;
- training state should be separated from layer parameters;
- forward propagation should explicitly return state required for backward propagation;
- optimizers should be independent from layers;
- interfaces should remain explicit and predictable;
- implementation details should remain hidden behind abstractions.

---

## Data Flow

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
- Train models on MNIST-style datasets
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
- Better testing infrastructure

---

## References

- Eigen: https://eigen.tuxfamily.org
- EigenRand: https://github.com/bab2min/EigenRand
- Sean Parent — Runtime Polymorphism:
  https://www.youtube.com/watch?v=QGcVXgEVMJg
- cppreference:
  https://en.cppreference.com

---

## Author

Nikolay Loginov  
HSE University — Software Engineering