#include "../fpu_case.hpp"

int main() {
  constexpr size_t N = 32;
  std::vector<uint32_t> a(N), b(N), c(N);
  std::mt19937 rng(3002);
  std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
  for (size_t i = 0; i < N; ++i) {
    a[i] = packBf16x2(dist(rng), dist(rng));
    b[i] = packBf16x2(dist(rng), dist(rng));
    c[i] = packBf16x2(dist(rng), dist(rng));
  }

  std::vector<uint32_t> output = runTernaryKernel<uint32_t>("kernel.cl", "vfma_bf16x2", a, b, c);

  int pass = 0;
  for (size_t i = 0; i < N; ++i) {
    float aLo = bf16BitsToFloat(uint16_t(a[i] & 0xffffu));
    float aHi = bf16BitsToFloat(uint16_t(a[i] >> 16));
    float bLo = bf16BitsToFloat(uint16_t(b[i] & 0xffffu));
    float bHi = bf16BitsToFloat(uint16_t(b[i] >> 16));
    float cLo = bf16BitsToFloat(uint16_t(c[i] & 0xffffu));
    float cHi = bf16BitsToFloat(uint16_t(c[i] >> 16));
    uint32_t ref = packBf16x2(aLo * bLo + cLo, aHi * bHi + cHi);
    pass += output[i] == ref;
  }
  std::printf("vfma_bf16x2: %d/%zu pass\n", pass, N);
  return pass == int(N) ? 0 : 1;
}
