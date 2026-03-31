#pragma once

#include "common.hpp"

#include <algorithm>

struct MMACaseConfig {
  const char *kernelPath;
  const char *kernelName;
  const char *caseName;
  size_t threads;
  size_t aRegsPerThread;
  size_t bRegsPerThread;
  size_t cRegsPerThread;
  size_t dRegsPerThread;
  float expectedValue;
  float tolerance;
};

inline int runMMACase(const MMACaseConfig &cfg) {
  std::vector<uint32_t> a(cfg.threads * cfg.aRegsPerThread,
                          packF16x2(1.0f, 1.0f));
  std::vector<uint32_t> b(cfg.threads * cfg.bRegsPerThread,
                          packF16x2(1.0f, 1.0f));
  std::vector<float> c(cfg.threads * cfg.cRegsPerThread, 0.0f);
  std::vector<float> d(cfg.threads * cfg.dRegsPerThread, 0.0f);

  auto ctx = createContext();
  cl_int err = CL_SUCCESS;
  cl_program program = buildProgram(ctx, cfg.kernelPath);
  cl_kernel kernel = clCreateKernel(program, cfg.kernelName, &err);
  CHECK_CL(err);

  cl_mem aBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * a.size(), a.data(), &err);
  CHECK_CL(err);
  cl_mem bBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * b.size(), b.data(), &err);
  CHECK_CL(err);
  cl_mem cBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * c.size(), c.data(), &err);
  CHECK_CL(err);
  cl_mem dBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                               sizeof(float) * d.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(aBuf), &aBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(bBuf), &bBuf));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cBuf), &cBuf));
  CHECK_CL(clSetKernelArg(kernel, 3, sizeof(dBuf), &dBuf));

  size_t global = cfg.threads;
  size_t local = cfg.threads;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local,
                                  0, nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, dBuf, CL_TRUE, 0, sizeof(float) * d.size(),
                               d.data(), 0, nullptr, nullptr));

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (float x : d) {
    float absErr = std::fabs(x - cfg.expectedValue);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < cfg.tolerance;
  }

  std::printf("%s: %d/%zu pass, maxAbsErr=%e\n", cfg.caseName, pass, d.size(),
              maxAbsErr);

  clReleaseMemObject(aBuf);
  clReleaseMemObject(bBuf);
  clReleaseMemObject(cBuf);
  clReleaseMemObject(dBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return pass == int(d.size()) ? 0 : 1;
}
