#include "bmm_case.hpp"

#include <cstdio>

int main() {
  using namespace bmm;

  BMMConfig cfg{};
  cfg.experts = 3;
  cfg.tokens = 8;
  cfg.k = 4;
  cfg.n = 8;
  cfg.splitKSlices = 2;

  std::vector<int> route = {1, 0, 2, 1, 0, 1, 2, 0};
  auto meta = buildRoutingMeta(cfg, route);

  if (meta.counts != std::vector<int>({3, 3, 2})) {
    std::fprintf(stderr, "unexpected expert counts\n");
    return 1;
  }
  if (meta.offsets != std::vector<int>({0, 16, 32, 48})) {
    std::fprintf(stderr, "unexpected padded offsets\n");
    return 1;
  }
  if (meta.totalGroupedTokens != 48) {
    std::fprintf(stderr, "unexpected grouped token span\n");
    return 1;
  }

  std::vector<float> x(cfg.tokens * cfg.k);
  for (int t = 0; t < cfg.tokens; ++t)
    for (int kk = 0; kk < cfg.k; ++kk)
      x[t * cfg.k + kk] = float(t * 10 + kk);

  auto grouped = scatterByExpert(x, cfg.tokens, cfg.k, meta);
  std::vector<float> gathered(cfg.tokens * cfg.k, -1.0f);
  gatherByToken(grouped, gathered, cfg.k, meta);

  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != gathered[i]) {
      std::fprintf(stderr, "scatter/gather mismatch at %zu\n", i);
      return 1;
    }
  }

  auto splitK = buildSplitKRanges(cfg.k, cfg.splitKSlices);
  if (splitK.size() != 2 || splitK[0].kBegin != 0 || splitK[0].kEnd != 2 ||
      splitK[1].kBegin != 2 || splitK[1].kEnd != 4) {
    std::fprintf(stderr, "unexpected split-k ranges\n");
    return 1;
  }

  BMMConfig perfCfg{};
  perfCfg.tokens = 16;
  perfCfg.k = 32;
  perfCfg.n = 96;
  double flops = effectiveFlops(perfCfg);
  if (std::llround(flops) != 98304) {
    std::fprintf(stderr, "unexpected effective flops\n");
    return 1;
  }
  double gflops = gflopsFromNs(flops, 1000000.0);
  if (std::fabs(gflops - 0.098304) > 1e-9) {
    std::fprintf(stderr, "unexpected gflops conversion\n");
    return 1;
  }

  std::puts("bmm_case_test: pass");
  return 0;
}
