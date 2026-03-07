#include <iostream>
#include <verify/location.hpp>

int main() {
  auto loc = NN_LOCATION();
  std::cout << loc.file << ":" << loc.line << " " << loc.func << "\n";

  return 0;
}