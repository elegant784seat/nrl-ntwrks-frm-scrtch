#include <iostream>
#include <verify/location.hpp>

int main() {
  auto loc = NN_LOC();
  std::cout << loc.file << ":" << loc.line << " " << loc.func << "\n";

  return 0;
}