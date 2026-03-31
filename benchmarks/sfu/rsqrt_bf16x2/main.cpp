#include "../common.hpp"

int main() {
  constexpr size_t N = 32;
  std::vector<uint32_t> input(N), output(N);
  std::vector<float> inLo(N), inHi(N);
  std::mt19937 rng(789);
  std::uniform_real_distribution<float> dist(0.5f, 4.0f);
  for (size_t i = 0; i < N; ++i) {
    inLo[i] = dist(rng);
    inHi[i] = dist(rng);
    input[i] = packBf16x2(inLo[i], inHi[i]);
  }

  auto ctx = createContext();
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, "rsqrt_bf16x2", &err);
  CHECK_CL(err);

  cl_mem inBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(uint32_t) * N, input.data(), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(uint32_t) * N, nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(inBuf), &inBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf));

  size_t global = N;
  size_t local = N;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global,
                                  &local, 0, nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0,
                               sizeof(uint32_t) * N, output.data(), 0, nullptr,
                               nullptr));

  int pass = 0;
  float maxRelErr = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    float outLo = bf16BitsToFloat(uint16_t(output[i] & 0xffffu));
    float outHi = bf16BitsToFloat(uint16_t(output[i] >> 16));
    float refLo = 1.0f / std::sqrt(inLo[i]);
    float refHi = 1.0f / std::sqrt(inHi[i]);
    float errLo = std::fabs(outLo - refLo) / std::max(std::fabs(refLo), 1e-6f);
    float errHi = std::fabs(outHi - refHi) / std::max(std::fabs(refHi), 1e-6f);
    maxRelErr = std::max(maxRelErr, std::max(errLo, errHi));
    pass += (errLo < 1e-2f);
    pass += (errHi < 1e-2f);
  }

  std::printf("rsqrt_bf16x2: %d/%zu lane-pass, maxRelErr=%e\n", pass, N * 2,
              maxRelErr);
  bool ok = (pass == int(N * 2));

  clReleaseMemObject(inBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return ok ? 0 : 1;
}
