#include "../fpu_case.hpp"

int main() {
  constexpr size_t N = 32;
  std::vector<uint32_t> a(N), b(N);
  std::mt19937 rng(2002);
  std::uniform_real_distribution<float> dist(0.5f, 2.0f);
  for (size_t i = 0; i < N; ++i) {
    a[i] = packF16x2(dist(rng), dist(rng));
    b[i] = packF16x2(dist(rng), dist(rng));
  }

  std::vector<uint32_t> output = runBinaryKernel<uint32_t>("kernel.cl", "vmul_f16x2", a, b);

  int pass = 0;
  for (size_t i = 0; i < N; ++i) {
    float aLo = halfBitsToFloat(uint16_t(a[i] & 0xffffu));
    float aHi = halfBitsToFloat(uint16_t(a[i] >> 16));
    float bLo = halfBitsToFloat(uint16_t(b[i] & 0xffffu));
    float bHi = halfBitsToFloat(uint16_t(b[i] >> 16));
    uint32_t ref = packF16x2(aLo * bLo, aHi * bHi);
    pass += output[i] == ref;
  }
  std::printf("vmul_f16x2: %d/%zu pass\n", pass, N);
  return pass == int(N) ? 0 : 1;
}
