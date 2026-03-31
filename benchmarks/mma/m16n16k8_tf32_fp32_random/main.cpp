#include "../common.hpp"

#include <algorithm>
#include <random>
#include <vector>

namespace {

constexpr int WarpThreads = 32;
constexpr int M = 16;
constexpr int N = 16;
constexpr int K = 8;
constexpr int ARegsPerThread = 4;
constexpr int BRegsPerThread = 4;
constexpr int CRegsPerThread = 8;

uint32_t tf32QuantizeBits(uint32_t bits) { return bits & 0xFFFFE000u; }

float tf32QuantizeFloat(float value) {
  return bitsToFloat(tf32QuantizeBits(floatToBits(value)));
}

std::vector<uint32_t> packATileRowMajorTF32(const std::vector<float> &mat) {
  std::vector<uint32_t> out(WarpThreads * ARegsPerThread);
  for (int reg = 0; reg < ARegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx = reg * WarpThreads + lane;
      int m = idx / K;
      int k = idx % K;
      out[lane * ARegsPerThread + reg] = tf32QuantizeBits(floatToBits(mat[m * K + k]));
    }
  }
  return out;
}

std::vector<uint32_t> packBTileColMajorTF32(const std::vector<float> &mat) {
  std::vector<uint32_t> out(WarpThreads * BRegsPerThread);
  for (int reg = 0; reg < BRegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx = reg * WarpThreads + lane;
      int k = idx / N;
      int n = idx % N;
      out[lane * BRegsPerThread + reg] = tf32QuantizeBits(floatToBits(mat[k * N + n]));
    }
  }
  return out;
}

std::vector<float> packCTileFp32(const std::vector<float> &mat) {
  std::vector<float> tile(WarpThreads * CRegsPerThread);
  for (int reg = 0; reg < CRegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx = reg * WarpThreads + lane;
      int m = idx / N;
      int n = idx % N;
      tile[lane * CRegsPerThread + reg] = mat[m * N + n];
    }
  }
  return tile;
}

void unpackCTileFp32(const std::vector<float> &tile, std::vector<float> &mat) {
  for (int reg = 0; reg < CRegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx = reg * WarpThreads + lane;
      int m = idx / N;
      int n = idx % N;
      mat[m * N + n] = tile[lane * CRegsPerThread + reg];
    }
  }
}

void matmulAccumulateRef(const std::vector<float> &a,
                         const std::vector<float> &b,
                         const std::vector<float> &c,
                         std::vector<float> &out) {
  out = c;
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      float lhs = a[m * K + k];
      for (int n = 0; n < N; ++n)
        out[m * N + n] += lhs * b[k * N + n];
    }
  }
}

} // namespace

int main() {
  std::mt19937 rng(20260316);
  std::uniform_real_distribution<float> dist(-1.25f, 1.25f);

  std::vector<float> a(M * K), b(K * N), c(M * N);
  for (auto &v : a) v = dist(rng);
  for (auto &v : b) v = dist(rng);
  for (auto &v : c) v = dist(rng);

  std::vector<float> aQ(a.size()), bQ(b.size());
  std::transform(a.begin(), a.end(), aQ.begin(), tf32QuantizeFloat);
  std::transform(b.begin(), b.end(), bQ.begin(), tf32QuantizeFloat);

  std::vector<float> ref(M * N);
  matmulAccumulateRef(aQ, bQ, c, ref);

  std::vector<uint32_t> aPacked = packATileRowMajorTF32(a);
  std::vector<uint32_t> bPacked = packBTileColMajorTF32(b);
  std::vector<float> cPacked = packCTileFp32(c);
  std::vector<float> dPacked(WarpThreads * CRegsPerThread, 0.0f);

  auto ctx = createContext();
  cl_int err = CL_SUCCESS;
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_kernel kernel = clCreateKernel(program, "mma_m16n16k8_tf32_fp32_random", &err);
  CHECK_CL(err);

  cl_mem aBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * aPacked.size(), aPacked.data(), &err);
  CHECK_CL(err);
  cl_mem bBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * bPacked.size(), bPacked.data(), &err);
  CHECK_CL(err);
  cl_mem cBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * cPacked.size(), cPacked.data(), &err);
  CHECK_CL(err);
  cl_mem dBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                               sizeof(float) * dPacked.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(aBuf), &aBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(bBuf), &bBuf));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cBuf), &cBuf));
  CHECK_CL(clSetKernelArg(kernel, 3, sizeof(dBuf), &dBuf));

  size_t global = WarpThreads;
  size_t local = WarpThreads;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local,
                                  0, nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, dBuf, CL_TRUE, 0, sizeof(float) * dPacked.size(),
                               dPacked.data(), 0, nullptr, nullptr));

  std::vector<float> got(M * N, 0.0f);
  unpackCTileFp32(dPacked, got);

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    float absErr = std::fabs(got[i] - ref[i]);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < 5e-3f;
  }

  std::printf("mma_m16n16k8_tf32_fp32_random: %d/%zu pass, maxAbsErr=%e\n",
              pass, got.size(), maxAbsErr);

  clReleaseMemObject(aBuf);
  clReleaseMemObject(bBuf);
  clReleaseMemObject(cBuf);
  clReleaseMemObject(dBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return pass == static_cast<int>(got.size()) ? 0 : 1;
}
