# Neural Networks from Scratch

C++ library for building and training fully connected neural networks from scratch.

Project name in Russian: **Нейросети с нуля**.

---

## Overview

This project is an educational neural network library written in modern C++.

The main goal is to implement core neural network components manually without using high-level machine learning frameworks such as PyTorch or TensorFlow.

The library is designed to help understand:

- forward propagation;
- backpropagation;
- gradient descent;
- optimizers;
- neural network architecture;
- type erasure in C++;
- clean API design.

---

## Features

- Fully connected neural networks
- Mini-batch training
- Forward / backward propagation
- MNIST-style datasets
- Modular architecture
- Type-erased interfaces
- Modern C++20 codebase
- CMake build system

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
- Dataset loaders
- Batch iteration
- Optional shuffling

---

## Tech Stack

- C++20
- CMake
- Eigen
- EigenRand
- clang-format
- clang-tidy

---

## Project Structure

```text
.
├── app/                  # Demo application and manual tests
├── nn/
│   ├── activation/       # Activation functions
│   ├── datasets/         # Dataset loaders
│   ├── layers/           # Neural network layers
│   ├── loss/             # Loss functions
│   ├── optimizer/        # Optimizers
│   ├── train/            # Training loop
│   └── Linalg.hpp        # Linear algebra aliases
├── external/             # Third-party libraries
├── data/                 # Dataset files
├── CMakeLists.txt
└── README.md
```

---

## Build

### Release

```bash
git clone https://github.com/elegant784seat/nrl-ntwrks-frm-scrtch.git
cd nrl-ntwrks-frm-scrtch

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

```bash
./cmake-build-release/app
```

or

```bash
cmake --build cmake-build-release --target app
```

---

## Example

```cpp
#include "layers/linear_layer.hpp"
#include "layers/nonlinear_layer.hpp"
#include "activation/relu_func.hpp"
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

## Architecture

The project actively uses:

- value semantics;
- RAII;
- type erasure;
- compile-time abstractions;
- modular neural network components.

Main abstractions:

- `AnyFunc`
- `AnyLayer`
- `AnyLoss`
- `AnyOptimizer`

The architecture is inspired by modern C++ design approaches and type erasure techniques discussed by [Sean Parent](chatgpt://generic-entity?number=0).

---

## Goals

- Understand neural networks from first principles
- Learn modern C++ architecture
- Build reusable ML abstractions
- Implement training pipeline manually
- Achieve high accuracy on MNIST-style datasets
- Create a clean educational codebase

---

## Future Plans

- CNN layers
- Serialization
- GPU support
- Python bindings
- More optimizers
- More datasets
- Parallel training

---

## References

-  [oai_citation:0‡eigen.tuxfamily.org](https://eigen.tuxfamily.org?utm_source=chatgpt.com)
-  [oai_citation:1‡github.com](https://github.com/bab2min/EigenRand?utm_source=chatgpt.com)
-  [oai_citation:2‡youtube.com](https://www.youtube.com/watch?v=QGcVXgEVMJg&utm_source=chatgpt.com)
-  [oai_citation:3‡en.cppreference.com](https://en.cppreference.com/w/?utm_source=chatgpt.com)

---

## Author

Nikolay Loginov  
HSE University — Software Engineering