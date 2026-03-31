#include "../workload_case.hpp"

namespace {
constexpr int M = 16;
constexpr int D = 96;
constexpr int H = 384;
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
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> x(M * D), w1(D * H), w2(H * D);
  for (auto &v : x) v = dist(rng);
  for (auto &v : w1) v = dist(rng);
  for (auto &v : w2) v = dist(rng);

  std::vector<uint32_t> xTiles;
  std::vector<uint32_t> w1Tiles;
  std::vector<uint32_t> w2Tiles;
  xTiles.reserve(KTiles * aTileWords());
  w1Tiles.reserve(HTiles * KTiles * bTileWords());
  w2Tiles.reserve(HTiles * OTiles * bTileWords());

  for (int k = 0; k < KTiles; ++k) {
    auto tile = packATileRowMajorFp16(x, M, D, 0, k * 16);
    xTiles.insert(xTiles.end(), tile.begin(), tile.end());
  }
  for (int h = 0; h < HTiles; ++h) {
    for (int k = 0; k < KTiles; ++k) {
      auto tile = packBTileColMajorFp16(w1, D, H, k * 16, h * 16);
      w1Tiles.insert(w1Tiles.end(), tile.begin(), tile.end());
    }
  }
  for (int h = 0; h < HTiles; ++h) {
    for (int o = 0; o < OTiles; ++o) {
      auto tile = packBTileColMajorFp16(w2, H, D, h * 16, o * 16);
      w2Tiles.insert(w2Tiles.end(), tile.begin(), tile.end());
    }
  }

  std::vector<float> hidden(M * H), activated(M * H), ref(M * D);
  matmulRef(x, w1, hidden, M, D, H);
  for (size_t i = 0; i < hidden.size(); ++i)
    activated[i] = geluApproxRef(hidden[i]);
  matmulRef(activated, w2, ref, M, H, D);

  auto ctx = createContext();
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, "gelu_mlp_fp16", &err);
  CHECK_CL(err);

  std::vector<float> outTiles(OTiles * cdTileScalars(), 0.0f);
  cl_mem xBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * xTiles.size(), xTiles.data(), &err);
  CHECK_CL(err);
  cl_mem w1Buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(uint32_t) * w1Tiles.size(), w1Tiles.data(), &err);
  CHECK_CL(err);
  cl_mem w2Buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(uint32_t) * w2Tiles.size(), w2Tiles.data(), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * outTiles.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(xBuf), &xBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(w1Buf), &w1Buf));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(w2Buf), &w2Buf));
  CHECK_CL(clSetKernelArg(kernel, 3, sizeof(outBuf), &outBuf));

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
  std::printf("gelu_mlp_fp16: %d/%zu pass, maxAbsErr=%e\n", pass, out.size(), maxAbsErr);

  clReleaseMemObject(xBuf);
  clReleaseMemObject(w1Buf);
  clReleaseMemObject(w2Buf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return pass == int(out.size()) ? 0 : 1;
}
