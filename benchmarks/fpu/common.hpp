#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_CL(expr)                                                         \
  do {                                                                         \
    cl_int _err = (expr);                                                      \
    if (_err != CL_SUCCESS) {                                                  \
      std::fprintf(stderr, "OpenCL error %d at %s:%d\n", _err, __FILE__,     \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

inline std::string readTextFile(const char *path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) {
    std::perror(path);
    std::exit(1);
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

struct OpenCLContext {
  cl_platform_id platform = nullptr;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
};

inline OpenCLContext createContext() {
  OpenCLContext ctx;
  CHECK_CL(clGetPlatformIDs(1, &ctx.platform, nullptr));
  CHECK_CL(clGetDeviceIDs(ctx.platform, CL_DEVICE_TYPE_DEFAULT, 1, &ctx.device,
                          nullptr));
  ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, nullptr);
  ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, nullptr);
  return ctx;
}

inline cl_program buildProgram(const OpenCLContext &ctx, const char *kernelPath) {
  std::string src = readTextFile(kernelPath);
  const char *srcPtr = src.c_str();
  size_t size = src.size();
  cl_int err = CL_SUCCESS;
  cl_program program =
      clCreateProgramWithSource(ctx.context, 1, &srcPtr, &size, &err);
  CHECK_CL(err);
  err = clBuildProgram(program, 1, &ctx.device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t logSize = 0;
    clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &logSize);
    std::vector<char> log(logSize + 1, 0);
    clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, logSize,
                          log.data(), nullptr);
    std::fprintf(stderr, "%s\n", log.data());
    CHECK_CL(err);
  }
  return program;
}

inline uint32_t floatToBits(float x) {
  uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

inline float bitsToFloat(uint32_t bits) {
  float x = 0.0f;
  std::memcpy(&x, &bits, sizeof(x));
  return x;
}

inline uint16_t floatToHalfBits(float value) {
  uint32_t bits = floatToBits(value);
  uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t exp = (bits >> 23) & 0xffu;
  uint32_t mant = bits & 0x7fffffu;

  if (exp == 0xffu) {
    if (mant == 0)
      return uint16_t(sign | 0x7c00u);
    uint32_t payload = mant >> 13;
    return uint16_t(sign | 0x7c00u | (payload ? payload : 1u));
  }

  int32_t halfExp = int32_t(exp) - 127 + 15;
  if (halfExp >= 31)
    return uint16_t(sign | 0x7c00u);

  if (halfExp <= 0) {
    if (halfExp < -10)
      return uint16_t(sign);

    mant |= 0x800000u;
    uint32_t shift = uint32_t(14 - halfExp);
    uint32_t rounded = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1u);
    uint32_t halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (rounded & 1u)))
      ++rounded;
    return uint16_t(sign | rounded);
  }

  uint32_t rounded = (uint32_t(halfExp) << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (rounded & 1u)))
    ++rounded;
  return uint16_t(sign | rounded);
}

inline float halfBitsToFloat(uint16_t bits) {
  uint32_t sign = (bits >> 15) & 0x1;
  uint32_t exp = (bits >> 10) & 0x1f;
  uint32_t mant = bits & 0x3ff;
  uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ff;
      out = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    out = (sign << 31) | 0x7f800000 | (mant << 13);
  } else {
    out = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return bitsToFloat(out);
}

inline uint16_t floatToBf16Bits(float value) {
  uint32_t bits = floatToBits(value);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t roundingBias = 0x7fffu + lsb;
  return uint16_t((bits + roundingBias) >> 16);
}

inline float bf16BitsToFloat(uint16_t bits) {
  return bitsToFloat(uint32_t(bits) << 16);
}

inline uint32_t packF16x2(float lo, float hi) {
  return uint32_t(floatToHalfBits(lo)) | (uint32_t(floatToHalfBits(hi)) << 16);
}

inline uint32_t packBf16x2(float lo, float hi) {
  return uint32_t(floatToBf16Bits(lo)) | (uint32_t(floatToBf16Bits(hi)) << 16);
}
