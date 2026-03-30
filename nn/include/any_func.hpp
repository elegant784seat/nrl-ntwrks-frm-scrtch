//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once

#include <memory>

#include "../include/verify/verify.hpp"
#include "tensor.hpp"

namespace nn {
class AnyFunc {
 public:
  AnyFunc() = delete;
  template <class Func>
  AnyFunc(Func func) : impl_(std::make_unique<Model<std::decay_t<Func>>>(std::move(func))) {}

  AnyFunc (const AnyFunc& other) : impl_(other.impl_->clone()) {
    NN_VERIFY(impl_ != nullptr);
  }
  AnyFunc& operator=(const AnyFunc& other) {
    if (&other == this) {
      return *this;
    }
    AnyFunc tmp(other);
    swap(tmp);
    return *this;
  }
  AnyFunc (AnyFunc&& other) noexcept = default;
  AnyFunc& operator=(AnyFunc&& other) noexcept = default;
  ~AnyFunc() = default;

  Tensor operator()(const Tensor& input) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->apply(input);
  }
  Tensor derivative(const Tensor& input, const Tensor& output) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->derivative(input, output);
  }
  void swap(AnyFunc& other) noexcept { std::swap(impl_, other.impl_); }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual Tensor apply(const Tensor& input) const = 0;
    virtual Tensor derivative(const Tensor& input, const Tensor& output) const = 0;
    virtual std::unique_ptr<Concept> clone() const = 0;
  };
  template <class Func>
  class Model final : public Concept {
   public:
    explicit Model(const Func& func) : func_(func) {}
    explicit Model(Func&& func) : func_(std::move(func)) {}

    std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<Func>>(func_); }
    Tensor apply(const Tensor& input) const override { return func_(input); }
    Tensor derivative(const Tensor& input, const Tensor& output) const override {
      return func_.derivative(input, output);
    }

   private:
    Func func_;
  };
  std::unique_ptr<Concept> impl_;
};

inline void swap(AnyFunc& left, AnyFunc& right) noexcept { left.swap(right); }
}  // namespace nn
