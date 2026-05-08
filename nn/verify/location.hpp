//
// Created by Loginov Nikolay on 22.02.2026.
//
#pragma once

#include <cstdint>

#if defined(_MSC_VER) && _MSC_VER <= 1900
#define __func__ __FUNCTION__
#endif

namespace nn {
struct Location {
  const char* func;
  const char* file;
  uint32_t line;
};
}  // namespace nn

#define NN_LOCATION() \
  ::nn::Location { __func__, __FILE__, static_cast<uint32_t>(__LINE__) }
