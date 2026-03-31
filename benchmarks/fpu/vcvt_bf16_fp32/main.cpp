#include "../fpu_case.hpp"

int main() {
  constexpr size_t N = 32;
  std::vector<float> input(N);
  std::mt19937 rng(1004);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (auto &x : input)
    x = dist(rng);

  std::vector<uint32_t> output = runUnaryKernel<float, uint32_t>("kernel.cl", "vcvt_bf16_fp32", input);

  int pass = 0;
  for (size_t i = 0; i < N; ++i)
    pass += uint16_t(output[i] & 0xffffu) == floatToBf16Bits(input[i]);
  std::printf("vcvt_bf16_fp32: %d/%zu pass\n", pass, N);
  return pass == int(N) ? 0 : 1;
}
