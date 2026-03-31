#pragma once

#include "common.hpp"

template <typename InT, typename OutT>
std::vector<OutT> runUnaryKernel(const char *kernelPath, const char *kernelName,
                                 const std::vector<InT> &input) {
  auto ctx = createContext();
  cl_program program = buildProgram(ctx, kernelPath);
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, kernelName, &err);
  CHECK_CL(err);

  std::vector<OutT> output(input.size());
  cl_mem inBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(InT) * input.size(), const_cast<InT *>(input.data()), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(OutT) * output.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(inBuf), &inBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(outBuf), &outBuf));

  size_t global = input.size();
  size_t local = input.size();
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local, 0,
                                  nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0, sizeof(OutT) * output.size(),
                               output.data(), 0, nullptr, nullptr));

  clReleaseMemObject(inBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return output;
}

template <typename T>
std::vector<T> runBinaryKernel(const char *kernelPath, const char *kernelName,
                               const std::vector<T> &lhs, const std::vector<T> &rhs) {
  auto ctx = createContext();
  cl_program program = buildProgram(ctx, kernelPath);
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, kernelName, &err);
  CHECK_CL(err);

  std::vector<T> output(lhs.size());
  cl_mem lhsBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(T) * lhs.size(), const_cast<T *>(lhs.data()), &err);
  CHECK_CL(err);
  cl_mem rhsBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(T) * rhs.size(), const_cast<T *>(rhs.data()), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(T) * output.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(lhsBuf), &lhsBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(rhsBuf), &rhsBuf));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(outBuf), &outBuf));

  size_t global = lhs.size();
  size_t local = lhs.size();
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local, 0,
                                  nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0, sizeof(T) * output.size(),
                               output.data(), 0, nullptr, nullptr));

  clReleaseMemObject(lhsBuf);
  clReleaseMemObject(rhsBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return output;
}

template <typename T>
std::vector<T> runTernaryKernel(const char *kernelPath, const char *kernelName,
                                const std::vector<T> &a, const std::vector<T> &b,
                                const std::vector<T> &c) {
  auto ctx = createContext();
  cl_program program = buildProgram(ctx, kernelPath);
  cl_int err = CL_SUCCESS;
  cl_kernel kernel = clCreateKernel(program, kernelName, &err);
  CHECK_CL(err);

  std::vector<T> output(a.size());
  cl_mem aBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(T) * a.size(), const_cast<T *>(a.data()), &err);
  CHECK_CL(err);
  cl_mem bBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(T) * b.size(), const_cast<T *>(b.data()), &err);
  CHECK_CL(err);
  cl_mem cBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(T) * c.size(), const_cast<T *>(c.data()), &err);
  CHECK_CL(err);
  cl_mem outBuf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY,
                                 sizeof(T) * output.size(), nullptr, &err);
  CHECK_CL(err);

  CHECK_CL(clSetKernelArg(kernel, 0, sizeof(aBuf), &aBuf));
  CHECK_CL(clSetKernelArg(kernel, 1, sizeof(bBuf), &bBuf));
  CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cBuf), &cBuf));
  CHECK_CL(clSetKernelArg(kernel, 3, sizeof(outBuf), &outBuf));

  size_t global = a.size();
  size_t local = a.size();
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, kernel, 1, nullptr, &global, &local, 0,
                                  nullptr, nullptr));
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, outBuf, CL_TRUE, 0, sizeof(T) * output.size(),
                               output.data(), 0, nullptr, nullptr));

  clReleaseMemObject(aBuf);
  clReleaseMemObject(bBuf);
  clReleaseMemObject(cBuf);
  clReleaseMemObject(outBuf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return output;
}
