#include "except.hpp"
#include "run_all_tests.hpp"
int main() {
  try {
    return nn::run_all_tests();
  } catch (...) {
    nn::react();
  }
}
