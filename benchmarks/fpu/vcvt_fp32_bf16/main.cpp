#include "../fpu_case.hpp"

int main() {
  constexpr size_t N = 32;
  std::vector<uint32_t> input(N);
  std::mt19937 rng(1003);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (size_t i = 0; i < N; ++i)
    input[i] = uint32_t(floatToBf16Bits(dist(rng)));

  std::vector<float> output = runUnaryKernel<uint32_t, float>("kernel.cl", "vcvt_fp32_bf16", input);

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float ref = bf16BitsToFloat(uint16_t(input[i] & 0xffffu));
    float absErr = std::fabs(output[i] - ref);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < 1e-6f;
  }
  std::printf("vcvt_fp32_bf16: %d/%zu pass, maxAbsErr=%e\n", pass, N, maxAbsErr);
  return pass == int(N) ? 0 : 1;
}
