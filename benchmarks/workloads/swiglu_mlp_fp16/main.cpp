#include "../workload_case.hpp"

namespace {
constexpr int M = 16;
constexpr int D = 96;
constexpr int H = 256;
constexpr int KTiles = D / 16;
constexpr int HTiles = H / 16;
constexpr int OTiles = D / 16;

void matmulRef(const std::vector<float> &a, const std::vector<float> &b,
               std::vector<float> &out, int m, int k, int n) {
  std::fill(out.begin(), out.end(), 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int kk = 0; kk < k; ++kk) {
      float lhs = a[i * k + kk];
      for (int j = 0; j < n; ++j)
        out[i * n + j] += lhs * b[kk * n + j];
    }
  }
}
}

int main() {
  std::mt19937 rng(54321);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> x(M * D), wg(D * H), wv(D * H), wo(H * D);
  for (auto &v : x) v = dist(rng);
  for (auto &v : wg) v = dist(rng);
  for (auto &v : wv) v = dist(rng);
  for (auto &v : wo) v = dist(rng);

  std::vector<uint32_t> xTiles;
  std::vector<uint32_t> gateTiles;
  std::vector<uint32_t> valueTiles;
  std::vector<uint32_t> outTilesPacked;
  xTiles.reserve(KTiles * aTileWords());
  gateTiles.reserve(HTiles * KTiles * bTileWords());
  valueTiles.reserve(HTiles * KTiles * bTileWords());
  outTilesPacked.reserve(HTiles * OTiles * bTileWords());

  for (int k = 0; k < KTiles; ++k) {
    auto tile = packATileRowMajorFp16(x, M, D, 0, k * 16);
    xTiles.insert(xTiles.end(), tile.begin(), tile.end());
  }
  for (int h = 0; h < HTiles; ++h) {
    for (int k = 0; k < KTiles; ++k) {
      auto gt = packBTileColMajorFp16(wg, D, H, k * 16, h * 16);
      auto vt = packBTileColMajorFp16(wv, D, H, k * 16, h * 16);
      gateTiles.insert(gateTiles.end(), gt.begin(), gt.end());
      valueTiles.insert(valueTiles.end(), vt.begin(), vt.end());
    }
  }
  for (int h = 0; h < HTiles; ++h) {
    for (int o = 0; o < OTiles; ++o) {
      auto tile = packBTileColMajorFp16(wo, H, D, h * 16, o * 16);
      outTilesPacked.insert(outTilesPacked.end(), tile.begin(), tile.end());
    }
  }

  std::vector<float> gate(M * H), value(M * H), fused(M * H), ref(M * D);
  matmulRef(x, wg, gate, M, D, H);
  matmulRef(x, wv, value, M, D, H);
  for (size_t i = 0; i < gate.size(); ++i)
    fused[i] = siluApproxRef(gate[i]) * value[i];
  matmulRef(fused, wo, ref, M, H, D);

  auto ctx = createContext();
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, "swiglu_mlp_fp16", &err);
  CHECK_CL(err);

  std::vector<float> outTiles(OTiles * cdTileScalars(), 0.0f);
  cl_mem xBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * xTiles.size(), xTiles.data(), &err);
  CHECK_CL(err);
  cl_mem gBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * gateTiles.size(), gateTiles.data(), &err);
  CHECK_CL(err);
  cl_mem vBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * valueTiles.size(), valueTiles.data(), &err);
  CHECK_CL(err);
  cl_mem oBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * outTilesPacked.size(), outTilesPacked.data(), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * outTiles.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(xBuf), &xBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(gBuf), &gBuf));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(vBuf), &vBuf));
  CHECK_CL(clSetKernelArg(kernel, 3, sizeof(oBuf), &oBuf));
  CHECK_CL(clSetKernelArg(kernel, 4, sizeof(outBuf), &outBuf));

  size_t global = WarpThreads;
  size_t local = WarpThreads;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local, 0,
                                  nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0, sizeof(float) * outTiles.size(),
                               outTiles.data(), 0, nullptr, nullptr));

  std::vector<float> out(M * D, 0.0f);
  for (int o = 0; o < OTiles; ++o) {
    std::vector<float> tile(outTiles.begin() + o * cdTileScalars(),
                            outTiles.begin() + (o + 1) * cdTileScalars());
    unpackCTileFp32(tile, out, D, 0, o * 16);
  }

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (size_t i = 0; i < out.size(); ++i) {
    float absErr = std::fabs(out[i] - ref[i]);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < 2e-1f;
  }
  std::printf("swiglu_mlp_fp16: %d/%zu pass, maxAbsErr=%e\n", pass, out.size(), maxAbsErr);

  clReleaseMemObject(xBuf);
  clReleaseMemObject(gBuf);
  clReleaseMemObject(vBuf);
  clReleaseMemObject(oBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return pass == int(out.size()) ? 0 : 1;
}
