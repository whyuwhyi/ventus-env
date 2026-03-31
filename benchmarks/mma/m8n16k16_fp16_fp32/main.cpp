#include "../mma_case.hpp"

int main() {
  return runMMACase({"kernel.cl", "mma_m8n16k16_fp16_fp32",
                     "mma_m8n16k16_fp16_fp32", 32, 2, 4, 4, 4, 16.0f,
                     1e-3f});
}
