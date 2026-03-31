#include "../common.hpp"

namespace {

inline float siluApproxRef(float x) {
  return x / (1.0f + std::exp(-x));
}

} // namespace

int main() {
  constexpr size_t N = 32;
  std::vector<float> input(N), output(N), ref(N);
  std::mt19937 rng(20260313);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  for (auto &x : input)
    x = dist(rng);

  auto ctx = createContext();
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, "silu_fp32", &err);
  CHECK_CL(err);

  cl_mem inBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * N, input.data(), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * N, nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(inBuf), &inBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf));

  size_t global = N;
  size_t local = N;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global,
                                  &local, 0, nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0, sizeof(float) * N,
                               output.data(), 0, nullptr, nullptr));

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    ref[i] = siluApproxRef(input[i]);
    float absErr = std::fabs(output[i] - ref[i]);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < 1e-4f;
  }

  std::printf("silu_fp32: %d/%zu pass, maxAbsErr=%e\n", pass, N, maxAbsErr);
  bool ok = (pass == int(N));

  clReleaseMemObject(inBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return ok ? 0 : 1;
}
