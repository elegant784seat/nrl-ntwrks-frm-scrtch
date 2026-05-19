//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once

#include <memory>

#include "../verify/verify.hpp"
#include "Linalg.hpp"

namespace nn {
class AnyFunc {
 public:
  AnyFunc() = default;
  template <class Func>
  AnyFunc(Func func) : impl_(std::make_unique<Model<std::decay_t<Func>>>(std::move(func))) {}

  AnyFunc(const AnyFunc& other) : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}
  AnyFunc& operator=(const AnyFunc& other) { return *this = AnyFunc(other); }

  AnyFunc(AnyFunc&& other) noexcept = default;
  AnyFunc& operator=(AnyFunc&& other) noexcept = default;
  ~AnyFunc() = default;

  Matrix operator()(const Matrix& input) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->operator()(input);
  }
  Matrix derivative(const Matrix& input, const Matrix& output) const {
    NN_VERIFY(impl_ != nullptr);
    return impl_->derivative(input, output);
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual Matrix operator()(const Matrix& input) const = 0;
    virtual Matrix derivative(const Matrix& input, const Matrix& output) const = 0;
    virtual std::unique_ptr<Concept> clone() const = 0;
  };
  template <class Func>
  class Model final : public Concept {
   public:
    explicit Model(const Func& func) : func_(func) {}
    explicit Model(Func&& func) : func_(std::move(func)) {}

    std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<Func>>(func_); }
    Matrix operator()(const Matrix& input) const override { return func_(input); }
    Matrix derivative(const Matrix& input, const Matrix& output) const override {
      return func_.derivative(input, output);
    }

   private:
    Func func_;
  };
  std::unique_ptr<Concept> impl_;
};
}  // namespace nn
