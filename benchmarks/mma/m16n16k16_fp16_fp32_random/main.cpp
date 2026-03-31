#include "../../workloads/workload_case.hpp"

namespace {

constexpr int Rows = 16;
constexpr int Cols = 16;
constexpr int K = 16;

std::vector<float> quantizeFp16(const std::vector<float> &src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = halfBitsToFloat(floatToHalfBits(src[i]));
  return dst;
}

std::vector<float> packCTileFp32(const std::vector<float> &mat,
                                 int cols, int rowBase, int colBase) {
  std::vector<float> tile(cdTileScalars());
  for (int reg = 0; reg < CDRegsPerThread; ++reg) {
    for (int lane = 0; lane < WarpThreads; ++lane) {
      int idx = reg * WarpThreads + lane;
      int m = idx / 16;
      int n = idx % 16;
      tile[lane * CDRegsPerThread + reg] =
          mat[(rowBase + m) * cols + (colBase + n)];
    }
  }
  return tile;
}

void matmulAccumulateRef(const std::vector<float> &a,
                         const std::vector<float> &b,
                         const std::vector<float> &c,
                         std::vector<float> &out) {
  out = c;
  for (int m = 0; m < Rows; ++m) {
    for (int k = 0; k < K; ++k) {
      float lhs = a[m * K + k];
      for (int n = 0; n < Cols; ++n)
        out[m * Cols + n] += lhs * b[k * Cols + n];
    }
  }
}

} // namespace

int main() {
  std::mt19937 rng(20260312);
  std::uniform_real_distribution<float> dist(-1.25f, 1.25f);

  std::vector<float> a(Rows * K), b(K * Cols), c(Rows * Cols);
  for (auto &v : a) v = dist(rng);
  for (auto &v : b) v = dist(rng);
  for (auto &v : c) v = dist(rng);

  std::vector<float> aQ = quantizeFp16(a);
  std::vector<float> bQ = quantizeFp16(b);
  std::vector<float> ref(Rows * Cols);
  matmulAccumulateRef(aQ, bQ, c, ref);

  std::vector<uint32_t> aPacked = packATileRowMajorFp16(a, Rows, K, 0, 0);
  std::vector<uint32_t> bPacked = packBTileColMajorFp16(b, K, Cols, 0, 0);
  std::vector<float> cPacked = packCTileFp32(c, Cols, 0, 0);
  std::vector<float> dPacked(cdTileScalars(), 0.0f);

  auto ctx = createContext();
  cl_int err = CL_SUCCESS;
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_kernel kernel = clCreateKernel(program, "mma_m16n16k16_fp16_fp32_random", &err);
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

  std::vector<float> got(Rows * Cols, 0.0f);
  unpackCTileFp32(dPacked, got, Cols, 0, 0);

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) {
    float absErr = std::fabs(got[i] - ref[i]);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < 1e-3f;
  }

  std::printf("mma_m16n16k16_fp16_fp32_random: %d/%zu pass, maxAbsErr=%e\n",
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
