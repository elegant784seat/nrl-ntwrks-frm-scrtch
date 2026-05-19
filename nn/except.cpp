#include "except.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

namespace nn {

[[noreturn]] void react() {
  try {
    throw;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << std::endl;

  } catch (...) {
    std::cerr << "Unknown exception" << std::endl;
  }
  std::exit(1);
}

}  // namespace nn
// Created by Loginov Nikolay on 20.04.2026.
//
