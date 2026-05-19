#include "app.hpp"
#include "except.hpp"
#include "run_all_tests.hpp"

int main() {
  try {
    return nn::app::RunMnistDemo();
  } catch (...) {
    nn::react();
  }
}
