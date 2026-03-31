#include "../common.hpp"

namespace {

std::vector<float> quantizeFp16Vec(const std::vector<float> &src) {
  std::vector<float> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = halfBitsToFloat(floatToHalfBits(src[i]));
  return dst;
}

inline float geluApproxRef(float x) {
  const float kAlpha = 0.7978845608f;
  const float z = kAlpha * (x + 0.044715f * x * x * x);
  const float expTerm = std::exp(2.0f * z);
  const float onePlusTanh = 2.0f - 2.0f / (expTerm + 1.0f);
  return 0.5f * x * onePlusTanh;
}

inline float siluApproxRef(float x) {
  return x / (1.0f + std::exp(-x));
}

std::vector<uint32_t> runUnaryKernelLocal(const std::vector<float> &input) {
  auto ctx = createContext();
  cl_int err = CL_SUCCESS;
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_kernel kernel = clCreateKernel(program, "vcvt_fp16_fp32_workloadlike", &err);
  CHECK_CL(err);

  std::vector<uint32_t> output(input.size());
  cl_mem inBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * input.size(),
                                const_cast<float *>(input.data()), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(uint32_t) * output.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(inBuf), &inBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf));

  size_t global = input.size();
  size_t local = 32;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local, 0,
                                  nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0,
                               sizeof(uint32_t) * output.size(), output.data(), 0,
                               nullptr, nullptr));

  clReleaseMemObject(inBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return output;
}

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

int check_case(const char *name, const std::vector<float> &input) {
  std::vector<uint32_t> output = runUnaryKernelLocal(input);

  int pass = 0;
  for (size_t i = 0; i < input.size(); ++i)
    pass += uint16_t(output[i] & 0xffffu) == floatToHalfBits(input[i]);
  std::printf("%s: %d/%zu pass\n", name, pass, input.size());
  return pass == static_cast<int>(input.size()) ? 0 : 1;
}

} // namespace

int main() {
  int rc = 0;

  {
    constexpr int M = 16;
    constexpr int D = 96;
    constexpr int H = 384;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(M * D), w1(D * H), hidden(M * H), act(M * H);
    for (auto &v : x) v = dist(rng);
    for (auto &v : w1) v = dist(rng);
    std::vector<float> xQ = quantizeFp16Vec(x);
    std::vector<float> w1Q = quantizeFp16Vec(w1);
    matmulRef(xQ, w1Q, hidden, M, D, H);
    for (size_t i = 0; i < hidden.size(); ++i)
      act[i] = geluApproxRef(hidden[i]);
    rc |= check_case("vcvt_fp16_fp32_gelu_path", act);
  }

  {
    constexpr int M = 16;
    constexpr int D = 96;
    constexpr int H = 256;
    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x(M * D), wg(D * H), wv(D * H), gate(M * H), value(M * H),
        fused(M * H);
    for (auto &v : x) v = dist(rng);
    for (auto &v : wg) v = dist(rng);
    for (auto &v : wv) v = dist(rng);
    std::vector<float> xQ = quantizeFp16Vec(x);
    std::vector<float> wgQ = quantizeFp16Vec(wg);
    std::vector<float> wvQ = quantizeFp16Vec(wv);
    matmulRef(xQ, wgQ, gate, M, D, H);
    matmulRef(xQ, wvQ, value, M, D, H);
    for (size_t i = 0; i < gate.size(); ++i)
      fused[i] = siluApproxRef(gate[i]) * value[i];
    rc |= check_case("vcvt_fp16_fp32_swiglu_path", fused);
  }

  return rc;
}
