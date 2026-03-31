#pragma once

#include "common.hpp"

#include <array>

constexpr int WarpThreads = 32;
constexpr int MTile = 16;
constexpr int NTile = 16;
constexpr int KTile = 16;
constexpr int ARegsPerThread = 4;
constexpr int BRegsPerThread = 4;
constexpr int CDRegsPerThread = 8;

inline size_t aTileWords() { return WarpThreads * ARegsPerThread; }
inline size_t bTileWords() { return WarpThreads * BRegsPerThread; }
inline size_t cdTileScalars() { return WarpThreads * CDRegsPerThread; }

inline std::vector<uint32_t> packATileRowMajorFp16(const std::vector<float> &mat,
                                                   int rows, int cols,
                                                   int rowBase, int colBase) {
  std::vector<uint32_t> out(aTileWords());
  for (int reg = 0; reg < ARegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx0 = reg * 64 + lane * 2;
      int idx1 = idx0 + 1;
      int m0 = idx0 / 16;
      int k0 = idx0 % 16;
      int m1 = idx1 / 16;
      int k1 = idx1 % 16;
      float lo = mat[(rowBase + m0) * cols + (colBase + k0)];
      float hi = mat[(rowBase + m1) * cols + (colBase + k1)];
      out[lane * ARegsPerThread + reg] = packF16x2(lo, hi);
    }
  }
  return out;
}

inline std::vector<uint32_t> packBTileColMajorFp16(const std::vector<float> &mat,
                                                   int rows, int cols,
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

inline void unpackCTileFp32(const std::vector<float> &tile,
                            std::vector<float> &mat,
                            int cols, int rowBase, int colBase) {
  for (int reg = 0; reg < CDRegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx = reg * WarpThreads + lane;
      int m = idx / 16;
      int n = idx % 16;
      mat[(rowBase + m) * cols + (colBase + n)] = tile[lane * CDRegsPerThread + reg];
    }
  }
}
