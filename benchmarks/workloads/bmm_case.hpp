#pragma once

#include "common.hpp"

#include <stdexcept>

namespace bmm {

constexpr int TileSize = 16;
constexpr int WarpThreads = 32;
constexpr int ARegsPerThread = 4;
constexpr int BRegsPerThread = 4;
constexpr int CRegsPerThread = 8;

struct BMMConfig {
  int experts = 2;
  int tokens = 16;
  int k = 32;
  int n = 96;
  int splitKSlices = 1;
};

struct RoutingMeta {
  std::vector<int> counts;
  std::vector<int> paddedCounts;
  std::vector<int> offsets;
  std::vector<int> tokenSlot;
  std::vector<int> tokenToGrouped;
  std::vector<int> groupedToToken;
  int totalGroupedTokens = 0;
};

struct SplitKRange {
  int kBegin = 0;
  int kEnd = 0;
};

struct BaseTask {
  int expert = 0;
  int groupedTokenBase = 0;
  int nTile = 0;
};

inline double effectiveFlops(const BMMConfig &cfg) {
  return 2.0 * double(cfg.tokens) * double(cfg.k) * double(cfg.n);
}

inline double gflopsFromNs(double flops, double ns) {
  if (ns <= 0.0)
    return 0.0;
  return flops / ns;
}

inline double msFromNs(double ns) { return ns / 1.0e6; }

inline int alignUp(int value, int align) {
  return ((value + align - 1) / align) * align;
}

inline size_t aTileWords() { return WarpThreads * ARegsPerThread; }
inline size_t bTileWords() { return WarpThreads * BRegsPerThread; }
inline size_t cdTileScalars() { return WarpThreads * CRegsPerThread; }

inline void require(bool cond, const char *msg) {
  if (!cond)
    throw std::runtime_error(msg);
}

inline BMMConfig readConfigFromEnv() {
  auto readInt = [](const char *name, int fallback) {
    if (const char *value = std::getenv(name))
      return std::max(1, std::atoi(value));
    return fallback;
  };

  BMMConfig cfg;
  cfg.experts = readInt("BMM_EXPERTS", cfg.experts);
  cfg.tokens = readInt("BMM_TOKENS", cfg.tokens);
  cfg.k = readInt("BMM_K", cfg.k);
  cfg.n = readInt("BMM_N", cfg.n);
  cfg.splitKSlices = readInt("BMM_SPLIT_K", cfg.splitKSlices);
  return cfg;
}

inline void validateConfig(const BMMConfig &cfg) {
  require(cfg.experts > 0, "experts must be positive");
  require(cfg.tokens > 0, "tokens must be positive");
  require(cfg.k > 0 && cfg.k % TileSize == 0, "k must be 16-aligned");
  require(cfg.n > 0 && cfg.n % TileSize == 0, "n must be 16-aligned");
  require(cfg.splitKSlices > 0, "split-k must be positive");
}

inline std::vector<SplitKRange> buildSplitKRanges(int extent, int slices) {
  require(extent > 0, "split-k extent must be positive");
  require(slices > 0, "split-k slices must be positive");
  slices = std::min(extent, slices);
  std::vector<SplitKRange> ranges;
  ranges.reserve(slices);
  int begin = 0;
  int base = extent / slices;
  int rem = extent % slices;
  for (int s = 0; s < slices; ++s) {
    int span = base + (s < rem ? 1 : 0);
    ranges.push_back({begin, begin + span});
    begin += span;
  }
  return ranges;
}

inline RoutingMeta buildRoutingMeta(const BMMConfig &cfg,
                                    const std::vector<int> &route) {
  require(int(route.size()) == cfg.tokens, "route size must match tokens");
  RoutingMeta meta;
  meta.counts.assign(cfg.experts, 0);
  for (int expert : route) {
    require(expert >= 0 && expert < cfg.experts, "route expert out of range");
    ++meta.counts[expert];
  }

  meta.paddedCounts.resize(cfg.experts, 0);
  meta.offsets.resize(cfg.experts + 1, 0);
  for (int e = 0; e < cfg.experts; ++e) {
    meta.paddedCounts[e] = alignUp(meta.counts[e], TileSize);
    meta.offsets[e + 1] = meta.offsets[e] + meta.paddedCounts[e];
  }
  meta.totalGroupedTokens = meta.offsets.back();

  meta.tokenSlot.assign(cfg.tokens, 0);
  meta.tokenToGrouped.assign(cfg.tokens, 0);
  meta.groupedToToken.assign(meta.totalGroupedTokens, -1);
  std::vector<int> running(cfg.experts, 0);
  for (int token = 0; token < cfg.tokens; ++token) {
    int expert = route[token];
    int slot = running[expert]++;
    int grouped = meta.offsets[expert] + slot;
    meta.tokenSlot[token] = slot;
    meta.tokenToGrouped[token] = grouped;
    meta.groupedToToken[grouped] = token;
  }
  return meta;
}

inline std::vector<float> scatterByExpert(const std::vector<float> &src,
                                          int rows, int cols,
                                          const RoutingMeta &meta) {
  require(int(src.size()) == rows * cols, "scatter src shape mismatch");
  std::vector<float> dst(meta.totalGroupedTokens * cols, 0.0f);
  for (int row = 0; row < rows; ++row) {
    int grouped = meta.tokenToGrouped[row];
    std::copy_n(src.data() + row * cols, cols, dst.data() + grouped * cols);
  }
  return dst;
}

inline void gatherByToken(const std::vector<float> &grouped,
                          std::vector<float> &dst, int cols,
                          const RoutingMeta &meta) {
  require(int(grouped.size()) == meta.totalGroupedTokens * cols,
          "gather grouped shape mismatch");
  require(int(dst.size()) == int(meta.tokenToGrouped.size()) * cols,
          "gather dst shape mismatch");
  std::fill(dst.begin(), dst.end(), 0.0f);
  for (int groupedRow = 0; groupedRow < meta.totalGroupedTokens; ++groupedRow) {
    int token = meta.groupedToToken[groupedRow];
    if (token < 0)
      continue;
    std::copy_n(grouped.data() + groupedRow * cols, cols,
                dst.data() + token * cols);
  }
}

inline std::vector<uint32_t> packFp16Rows(const std::vector<float> &src,
                                          int rows, int cols) {
  require(int(src.size()) == rows * cols, "pack src shape mismatch");
  require(cols % 2 == 0, "fp16 row packing requires even cols");
  std::vector<uint32_t> packed(rows * (cols / 2), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; col += 2) {
      packed[row * (cols / 2) + (col / 2)] =
          packF16x2(src[row * cols + col], src[row * cols + col + 1]);
    }
  }
  return packed;
}

inline std::vector<uint32_t>
packExpertWeightsFp16(const std::vector<float> &src, int experts, int k,
                      int n) {
  require(int(src.size()) == experts * k * n, "weight shape mismatch");
  require(n % 2 == 0, "fp16 weight packing requires even n");
  std::vector<uint32_t> packed(experts * k * (n / 2), 0);
  int rowWords = n / 2;
  for (int e = 0; e < experts; ++e) {
    for (int row = 0; row < k; ++row) {
      for (int col = 0; col < n; col += 2) {
        int srcBase = e * k * n + row * n + col;
        int dstBase = e * k * rowWords + row * rowWords + (col / 2);
        packed[dstBase] = packF16x2(src[srcBase], src[srcBase + 1]);
      }
    }
  }
  return packed;
}

inline std::vector<uint32_t>
packBWindowColMajorFp16(const std::vector<float> &mat, int rows, int cols,
                        int rowBase, int colBase) {
  std::vector<uint32_t> out(bTileWords());
  for (int reg = 0; reg < BRegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx0 = reg * 64 + lane * 2;
      int idx1 = idx0 + 1;
      int k0 = idx0 / 16;
      int n0 = idx0 % 16;
      int k1 = idx1 / 16;
      int n1 = idx1 % 16;
      float lo = mat[(rowBase + k0) * cols + (colBase + n0)];
      float hi = mat[(rowBase + k1) * cols + (colBase + n1)];
      out[lane * BRegsPerThread + reg] = packF16x2(lo, hi);
    }
  }
  return out;
}

inline std::vector<uint32_t>
buildResidentWeightTiles(const std::vector<float> &weights,
                         const BMMConfig &cfg) {
  require(int(weights.size()) == cfg.experts * cfg.k * cfg.n,
          "resident weight tile shape mismatch");
  int kTiles = cfg.k / TileSize;
  int nTiles = cfg.n / TileSize;
  std::vector<uint32_t> packed(cfg.experts * nTiles * kTiles * bTileWords(),
                               0u);
  for (int expert = 0; expert < cfg.experts; ++expert) {
    std::vector<float> expertMat(weights.begin() + expert * cfg.k * cfg.n,
                                 weights.begin() +
                                     (expert + 1) * cfg.k * cfg.n);
    for (int nTile = 0; nTile < nTiles; ++nTile) {
      for (int kTile = 0; kTile < kTiles; ++kTile) {
        auto tile = packBWindowColMajorFp16(expertMat, cfg.k, cfg.n,
                                            kTile * TileSize, nTile * TileSize);
        size_t base =
            ((expert * nTiles + nTile) * kTiles + kTile) * bTileWords();
        std::copy(tile.begin(), tile.end(), packed.begin() + base);
      }
    }
  }
  return packed;
}

inline std::vector<BaseTask> buildBaseTasks(const BMMConfig &cfg,
                                            const RoutingMeta &meta) {
  std::vector<BaseTask> tasks;
  int nTiles = cfg.n / TileSize;
  for (int expert = 0; expert < cfg.experts; ++expert) {
    for (int grouped = meta.offsets[expert]; grouped < meta.offsets[expert + 1];
         grouped += TileSize) {
      for (int nTile = 0; nTile < nTiles; ++nTile)
        tasks.push_back({expert, grouped, nTile});
    }
  }
  return tasks;
}

inline std::vector<float>
unpackTaskTilesFp32(const std::vector<float> &tiles,
                    const std::vector<BaseTask> &tasks, int totalGroupedTokens,
                    int n) {
  require(int(tiles.size()) == int(tasks.size()) * WarpThreads * CRegsPerThread,
          "tile buffer shape mismatch");
  std::vector<float> grouped(totalGroupedTokens * n, 0.0f);
  for (size_t taskIdx = 0; taskIdx < tasks.size(); ++taskIdx) {
    const BaseTask &task = tasks[taskIdx];
    for (int reg = 0; reg < CRegsPerThread; ++reg) {
      for (int lane = 0; lane < WarpThreads; ++lane) {
        int linear = reg * WarpThreads + lane;
        int row = task.groupedTokenBase + linear / TileSize;
        int col = task.nTile * TileSize + linear % TileSize;
        grouped[row * n + col] =
            tiles[(taskIdx * WarpThreads + lane) * CRegsPerThread + reg];
      }
    }
  }
  return grouped;
}

inline void groupedGemmRef(const std::vector<float> &groupedX,
                           const std::vector<float> &weights,
                           std::vector<float> &groupedY, const BMMConfig &cfg,
                           const RoutingMeta &meta) {
  require(int(groupedX.size()) == meta.totalGroupedTokens * cfg.k,
          "grouped x shape mismatch");
  require(int(weights.size()) == cfg.experts * cfg.k * cfg.n,
          "grouped gemm weight shape mismatch");
  groupedY.assign(meta.totalGroupedTokens * cfg.n, 0.0f);
  for (int expert = 0; expert < cfg.experts; ++expert) {
    for (int grouped = meta.offsets[expert]; grouped < meta.offsets[expert + 1];
         ++grouped) {
      for (int kk = 0; kk < cfg.k; ++kk) {
        float lhs = groupedX[grouped * cfg.k + kk];
        const float *wRow =
            weights.data() + expert * cfg.k * cfg.n + kk * cfg.n;
        float *outRow = groupedY.data() + grouped * cfg.n;
        for (int nn = 0; nn < cfg.n; ++nn)
          outRow[nn] += lhs * wRow[nn];
      }
    }
  }
}

inline void moeRef(const std::vector<float> &x,
                   const std::vector<float> &weights, std::vector<float> &y,
                   const BMMConfig &cfg, const std::vector<int> &route) {
  auto meta = buildRoutingMeta(cfg, route);
  auto groupedX = scatterByExpert(x, cfg.tokens, cfg.k, meta);
  std::vector<float> groupedY;
  groupedGemmRef(groupedX, weights, groupedY, cfg, meta);
  y.assign(cfg.tokens * cfg.n, 0.0f);
  gatherByToken(groupedY, y, cfg.n, meta);
}

} // namespace bmm
