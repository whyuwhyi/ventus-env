#include "../mma_case.hpp"

int main() {
  return runMMACase({"kernel.cl", "mma_m16n16k16_fp16_fp32",
                     "mma_m16n16k16_fp16_fp32", 32, 4, 4, 8, 8, 16.0f,
                     1e-3f});
}
