//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once
#include <sstream>
#include <stdexcept>
#include <string>
#include "../include/verify/location.hpp"

namespace nn::verify {
class VerifyError : public std::runtime_error {
public:
  VerifyError(const std::string& message) : std::runtime_error(message) {
  }
};

[[noreturn]] inline void
fail(nn::Location location, const std::string& message) {
  std::ostringstream oss;
  oss << "[NN_VERIFY] " << message
      << "\n  at " << location.file << ":" << location.line
      << " (" << location.func << ")";
  throw VerifyError(oss.str());
}

inline std::string make_message(const char* expression,
                                const std::string& message) {
  std::ostringstream oss;
  oss << "check failed: " << expression;
  if (!message.empty()) {
    oss << " | " << message;
  }
  return oss.str();
}
} // namespace nn::verify

#define NN_VERIFY(condition)                                                        \
  do {                                                                              \
    if (!(condition)) {                                                             \
      ::nn::verify::fail(NN_LOCATION(), ::nn::verify::make_message(#condition, ""));\
    }                                                                               \
  } while (0)

#define NN_VERIFY_MESSAGE(condition, message)                                           \
  do {                                                                              \
    if (!(condition)) {                                                             \
      ::nn::verify::fail(NN_LOCATION(), ::nn::verify::make_message(#condition, (message)));\
    }                                                                               \
  } while (0)

#ifndef NDEBUG
#define NN_ASSERT(condition) NN_VERIFY(condition)
#define NN_ASSERT_MESSAGE(condition, message) NN_VERIFY_MESSAGE(condition, message)
#else
#define NN_ASSERT(condition) do { (void)sizeof(condition); } while (0)
#define NN_ASSERT_MESSAGE(condition, message) do { (void)sizeof(condition); } while (0)
#endif