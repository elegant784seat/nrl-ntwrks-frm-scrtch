//
// Created by Loginov Nikolay on 07.05.2026.
//

#pragma once
#include <memory>

#include "verify/verify.hpp"

namespace nn {
class AnyValue {
 public:
  AnyValue() = default;

  template <class T>
  explicit AnyValue(T value) : impl_(std::make_unique<Model<std::decay_t<T>>>(std::move(value))) {}

  AnyValue(const AnyValue& other) : impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

  AnyValue& operator=(const AnyValue& other) { return *this = AnyValue(other); }

  AnyValue(AnyValue&& other) noexcept = default;
  AnyValue& operator=(AnyValue&& oter) noexcept = default;
  ~AnyValue() = default;

  bool hasValue() const noexcept { return impl_ != nullptr; }

  explicit operator bool() const noexcept { return hasValue(); }

  void reset() noexcept { impl_.reset(); }

  template <class T>
  bool hasType() const noexcept {
    using ValueType = std::decay_t<T>;
    return impl_ != nullptr &&
           impl_->type() == typeid(ValueType);  // я кстати обычно всегда писал не &&, а and, но
                                                // наверно в каком-то лютом случае это будет ub
  }

  template <class T>
  std::decay_t<T>& get() {
    using ValueType = std::decay_t<T>;
    NN_VERIFY(impl_ != nullptr);
    NN_VERIFY(impl_->type() == typeid(ValueType));
    return static_cast<Model<ValueType>&>(*impl_).value_;
  }

  template <class T>
  const std::decay_t<T>& get() const {
    using ValueType = std::decay_t<T>;
    NN_VERIFY(impl_ != nullptr);
    NN_VERIFY(impl_->type() == typeid(ValueType));
    return static_cast<const Model<ValueType>&>(*impl_).value_;
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
    virtual std::unique_ptr<Concept> clone() const = 0;
    virtual const std::type_info& type() const noexcept = 0;
  };
  template <class T>
  struct Model final : Concept {
    explicit Model(T value) : value_(std::move(value)) {}
    std::unique_ptr<Concept> clone() const override { return std::make_unique<Model<T>>(value_); }
    const std::type_info& type() const noexcept override { return typeid(T); }

    T value_;
  };
  std::unique_ptr<Concept> impl_;
};
}  // namespace nn
