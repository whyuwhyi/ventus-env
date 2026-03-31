#include "../fpu/fpu_case.hpp"

namespace {

int check_case(const char *name, const std::vector<uint32_t> &got,
               const std::vector<uint32_t> &ref) {
  int pass = 0;
  for (size_t i = 0; i < got.size(); ++i)
    pass += got[i] == ref[i];
  std::printf("%s: %d/%zu pass\n", name, pass, got.size());
  return pass == static_cast<int>(got.size()) ? 0 : 1;
}

} // namespace

int main() {
  constexpr size_t N = 32;
  std::vector<uint32_t> input(N);
  for (size_t i = 0; i < N; ++i)
    input[i] = 0x100u + static_cast<uint32_t>(i);

  int rc = 0;

  {
    auto out = runUnaryKernel<uint32_t, uint32_t>("kernel_idx.cl", "shuffle_idx_i32", input);
    std::vector<uint32_t> ref(N, input[3]);
    rc |= check_case("shuffle_idx_i32", out, ref);
  }

  {
    auto out = runUnaryKernel<uint32_t, uint32_t>("kernel_up.cl", "shuffle_up_i32", input);
    std::vector<uint32_t> ref(N);
    for (size_t i = 0; i < N; ++i)
      ref[i] = (i >= 1) ? input[i - 1] : input[i];
    rc |= check_case("shuffle_up_i32", out, ref);
  }

  {
    auto out = runUnaryKernel<uint32_t, uint32_t>("kernel_down.cl", "shuffle_down_i32", input);
    std::vector<uint32_t> ref(N);
    for (size_t i = 0; i < N; ++i)
      ref[i] = (i + 2 < N) ? input[i + 2] : input[i];
    rc |= check_case("shuffle_down_i32", out, ref);
  }

  {
    auto out = runUnaryKernel<uint32_t, uint32_t>("kernel_bfly.cl", "shuffle_bfly_i32", input);
    std::vector<uint32_t> ref(N);
    for (size_t i = 0; i < N; ++i)
      ref[i] = input[i ^ 4u];
    rc |= check_case("shuffle_bfly_i32", out, ref);
  }

  {
    auto out = runUnaryKernel<uint32_t, uint32_t>("kernel_masked.cl", "shuffle_down_i32_masked", input);
    std::vector<uint32_t> ref = input;
    for (size_t i = 0; i < 16; ++i)
      ref[i] = (i + 1 < 16) ? input[i + 1] : input[i];
    rc |= check_case("shuffle_down_i32_masked", out, ref);
  }

  return rc;
}
