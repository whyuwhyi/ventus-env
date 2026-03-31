#include "../bmm_case.hpp"

namespace {

std::vector<int> buildTop1Route(int tokens, int experts) {
  std::vector<int> route(tokens, 0);
  for (int token = 0; token < tokens; ++token) {
    if (token < experts)
      route[token] = token;
    else
      route[token] = (token * 7 + 3) % experts;
  }
  return route;
}

} // namespace

int main() {
  using namespace bmm;

  BMMConfig cfg = readConfigFromEnv();
  try {
    validateConfig(cfg);
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "invalid bmm config: %s\n", ex.what());
    return 1;
  }

  std::mt19937 rng(24680);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

  std::vector<float> x(cfg.tokens * cfg.k);
  std::vector<float> weights(cfg.experts * cfg.k * cfg.n);
  for (auto &v : x)
    v = dist(rng);
  for (auto &v : weights)
    v = dist(rng);

  std::vector<int> route = buildTop1Route(cfg.tokens, cfg.experts);
  RoutingMeta meta = buildRoutingMeta(cfg, route);
  std::vector<BaseTask> tasks = buildBaseTasks(cfg, meta);
  std::vector<SplitKRange> splitKTiles =
      buildSplitKRanges(cfg.k / TileSize, cfg.splitKSlices);
  cfg.splitKSlices = int(splitKTiles.size());

  std::vector<float> ref;
  moeRef(x, weights, ref, cfg, route);

  std::vector<uint32_t> xPacked = packFp16Rows(x, cfg.tokens, cfg.k);
  std::vector<uint32_t> bTiles = buildResidentWeightTiles(weights, cfg);
  int groupedTiles = meta.totalGroupedTokens / TileSize;
  int kTiles = cfg.k / TileSize;
  int nTiles = cfg.n / TileSize;
  std::vector<uint32_t> aTiles(groupedTiles * kTiles * aTileWords(), 0u);
  std::vector<int> groupedToToken(meta.totalGroupedTokens, -1);
  std::vector<int> taskExpert(tasks.size(), 0);
  std::vector<int> taskGroupedBase(tasks.size(), 0);
  std::vector<int> taskNTile(tasks.size(), 0);
  for (size_t i = 0; i < tasks.size(); ++i) {
    taskExpert[i] = tasks[i].expert;
    taskGroupedBase[i] = tasks[i].groupedTokenBase;
    taskNTile[i] = tasks[i].nTile;
  }

  std::vector<int> sliceBegin(splitKTiles.size(), 0);
  std::vector<int> sliceEnd(splitKTiles.size(), 0);
  for (size_t i = 0; i < splitKTiles.size(); ++i) {
    sliceBegin[i] = splitKTiles[i].kBegin;
    sliceEnd[i] = splitKTiles[i].kEnd;
  }

  std::vector<float> partialTiles(tasks.size() * cfg.splitKSlices * WarpThreads *
                                  CRegsPerThread,
                                  0.0f);
  std::vector<float> reducedTiles(tasks.size() * WarpThreads * CRegsPerThread,
                                  0.0f);

  cl_int err = CL_SUCCESS;
  auto ctx = createContext();
  clReleaseCommandQueue(ctx.queue);
  ctx.queue = clCreateCommandQueue(ctx.context, ctx.device,
                                   CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_CL(err);
  cl_program program = buildProgram(ctx, "kernel.cl");
  cl_kernel scatterKernel = clCreateKernel(program, "route_scatter_top1", &err);
  CHECK_CL(err);
  cl_kernel gemmKernel =
      clCreateKernel(program, "grouped_gemm_splitk_fp16", &err);
  CHECK_CL(err);
  cl_kernel reduceKernel = clCreateKernel(program, "reduce_splitk_fp32", &err);
  CHECK_CL(err);

  cl_mem xBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(uint32_t) * xPacked.size(), xPacked.data(),
                               &err);
  CHECK_CL(err);
  cl_mem routeBuf = clCreateBuffer(ctx.context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(int) * route.size(), route.data(), &err);
  CHECK_CL(err);
  cl_mem slotBuf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int) * meta.tokenSlot.size(),
                                  meta.tokenSlot.data(), &err);
  CHECK_CL(err);
  cl_mem offsetsBuf = clCreateBuffer(ctx.context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(int) * meta.offsets.size(),
                                     meta.offsets.data(), &err);
  CHECK_CL(err);
  cl_mem aTileBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(uint32_t) * aTiles.size(), aTiles.data(), &err);
  CHECK_CL(err);
  cl_mem groupedToTokenBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * groupedToToken.size(), groupedToToken.data(),
                     &err);
  CHECK_CL(err);
  cl_mem weightBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(uint32_t) * bTiles.size(), bTiles.data(), &err);
  CHECK_CL(err);
  cl_mem taskExpertBuf = clCreateBuffer(ctx.context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        sizeof(int) * taskExpert.size(),
                                        taskExpert.data(), &err);
  CHECK_CL(err);
  cl_mem taskGroupedBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * taskGroupedBase.size(), taskGroupedBase.data(),
                     &err);
  CHECK_CL(err);
  cl_mem taskNTileBuf = clCreateBuffer(ctx.context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(int) * taskNTile.size(),
                                       taskNTile.data(), &err);
  CHECK_CL(err);
  cl_mem sliceBeginBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(int) * sliceBegin.size(), sliceBegin.data(), &err);
  CHECK_CL(err);
  cl_mem sliceEndBuf = clCreateBuffer(ctx.context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(int) * sliceEnd.size(),
                                      sliceEnd.data(), &err);
  CHECK_CL(err);
  cl_mem partialBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * partialTiles.size(), partialTiles.data(),
                     &err);
  CHECK_CL(err);
  cl_mem reducedBuf =
      clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(float) * reducedTiles.size(), reducedTiles.data(),
                     &err);
  CHECK_CL(err);

  int xWords = cfg.k / 2;
  int baseTaskCount = int(tasks.size());
  int taskChunk = 0;
  if (const char *value = std::getenv("BMM_TASK_CHUNK"))
    taskChunk = std::max(1, std::atoi(value));
  bool debugSync = std::getenv("BMM_DEBUG_SYNC") != nullptr;
  double kernelNs = 0.0;
  auto e2eStart = std::chrono::steady_clock::now();

  auto syncStage = [&](const char *stage) {
    if (!debugSync)
      return;
    std::fprintf(stderr, "[bmm_fp16] stage=%s enqueue done\n", stage);
    CHECK_CL(clFinish(ctx.queue));
    std::fprintf(stderr, "[bmm_fp16] stage=%s finish ok\n", stage);
  };

  CHECK_CL(clSetKernelArg(scatterKernel, 0, sizeof(xBuf), &xBuf));
  CHECK_CL(clSetKernelArg(scatterKernel, 1, sizeof(routeBuf), &routeBuf));
  CHECK_CL(clSetKernelArg(scatterKernel, 2, sizeof(slotBuf), &slotBuf));
  CHECK_CL(clSetKernelArg(scatterKernel, 3, sizeof(offsetsBuf), &offsetsBuf));
  CHECK_CL(clSetKernelArg(scatterKernel, 4, sizeof(cfg.tokens), &cfg.tokens));
  CHECK_CL(clSetKernelArg(scatterKernel, 5, sizeof(xWords), &xWords));
  CHECK_CL(clSetKernelArg(scatterKernel, 6, sizeof(kTiles), &kTiles));
  CHECK_CL(clSetKernelArg(scatterKernel, 7, sizeof(aTileBuf), &aTileBuf));
  CHECK_CL(clSetKernelArg(scatterKernel, 8, sizeof(groupedToTokenBuf),
                          &groupedToTokenBuf));

  CHECK_CL(clSetKernelArg(gemmKernel, 0, sizeof(aTileBuf), &aTileBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 1, sizeof(weightBuf), &weightBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 2, sizeof(taskExpertBuf), &taskExpertBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 3, sizeof(taskGroupedBuf), &taskGroupedBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 4, sizeof(taskNTileBuf), &taskNTileBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 5, sizeof(sliceBeginBuf), &sliceBeginBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 6, sizeof(sliceEndBuf), &sliceEndBuf));
  CHECK_CL(clSetKernelArg(gemmKernel, 9, sizeof(cfg.splitKSlices),
                          &cfg.splitKSlices));
  CHECK_CL(clSetKernelArg(gemmKernel, 10, sizeof(kTiles), &kTiles));
  CHECK_CL(clSetKernelArg(gemmKernel, 11, sizeof(nTiles), &nTiles));
  CHECK_CL(clSetKernelArg(gemmKernel, 12, sizeof(partialBuf), &partialBuf));

  CHECK_CL(clSetKernelArg(reduceKernel, 0, sizeof(partialBuf), &partialBuf));
  CHECK_CL(clSetKernelArg(reduceKernel, 3, sizeof(cfg.splitKSlices),
                          &cfg.splitKSlices));
  CHECK_CL(clSetKernelArg(reduceKernel, 4, sizeof(reducedBuf), &reducedBuf));

  size_t scatterGlobal = alignUp(cfg.tokens * xWords, WarpThreads);
  size_t warpLocal = WarpThreads;
  if (debugSync) {
    std::fprintf(stderr,
                 "[bmm_fp16] cfg experts=%d tokens=%d k=%d n=%d splitK=%d "
                 "groupedTokens=%d groupedTiles=%d kTiles=%d nTiles=%d tasks=%d\n",
                 cfg.experts, cfg.tokens, cfg.k, cfg.n, cfg.splitKSlices,
                 meta.totalGroupedTokens, groupedTiles, kTiles, nTiles,
                 baseTaskCount);
    std::fprintf(stderr,
                 "[bmm_fp16] launch route_scatter_top1 blocks=%zu threads=%zu\n",
                 scatterGlobal / warpLocal, warpLocal);
  }
  cl_event scatterEvent = nullptr;
  CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, scatterKernel, 1, nullptr,
                                  &scatterGlobal, &warpLocal, 0, nullptr,
                                  &scatterEvent));
  syncStage("route_scatter_top1");
  CHECK_CL(clFinish(ctx.queue));
  kernelNs += eventDurationNs(scatterEvent);
  clReleaseEvent(scatterEvent);

  int launchChunk = taskChunk > 0 ? taskChunk : baseTaskCount;
  for (int taskOffset = 0; taskOffset < baseTaskCount; taskOffset += launchChunk) {
    int launchTaskCount = std::min(launchChunk, baseTaskCount - taskOffset);
    size_t gemmGlobal =
        size_t(launchTaskCount * cfg.splitKSlices * WarpThreads);
    CHECK_CL(clSetKernelArg(gemmKernel, 7, sizeof(taskOffset), &taskOffset));
    CHECK_CL(clSetKernelArg(gemmKernel, 8, sizeof(launchTaskCount),
                            &launchTaskCount));
    if (debugSync) {
      std::fprintf(stderr,
                   "[bmm_fp16] launch grouped_gemm_splitk_fp16 taskOffset=%d "
                   "taskCount=%d blocks=%zu threads=%zu\n",
                   taskOffset, launchTaskCount, gemmGlobal / warpLocal,
                   warpLocal);
    }
    cl_event event = nullptr;
    CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, gemmKernel, 1, nullptr,
                                    &gemmGlobal, &warpLocal, 0, nullptr,
                                    &event));
    syncStage("grouped_gemm_splitk_fp16");
    CHECK_CL(clFinish(ctx.queue));
    kernelNs += eventDurationNs(event);
    clReleaseEvent(event);
  }

  for (int taskOffset = 0; taskOffset < baseTaskCount; taskOffset += launchChunk) {
    int launchTaskCount = std::min(launchChunk, baseTaskCount - taskOffset);
    size_t reduceGlobal = size_t(launchTaskCount * WarpThreads);
    CHECK_CL(clSetKernelArg(reduceKernel, 1, sizeof(taskOffset), &taskOffset));
    CHECK_CL(clSetKernelArg(reduceKernel, 2, sizeof(launchTaskCount),
                            &launchTaskCount));
    if (debugSync) {
      std::fprintf(stderr,
                   "[bmm_fp16] launch reduce_splitk_fp32 taskOffset=%d "
                   "taskCount=%d blocks=%zu threads=%zu\n",
                   taskOffset, launchTaskCount, reduceGlobal / warpLocal,
                   warpLocal);
    }
    cl_event event = nullptr;
    CHECK_CL(clEnqueueNDRangeKernel(ctx.queue, reduceKernel, 1, nullptr,
                                    &reduceGlobal, &warpLocal, 0, nullptr,
                                    &event));
    syncStage("reduce_splitk_fp32");
    CHECK_CL(clFinish(ctx.queue));
    kernelNs += eventDurationNs(event);
    clReleaseEvent(event);
  }
  CHECK_CL(clFinish(ctx.queue));
  CHECK_CL(clEnqueueReadBuffer(ctx.queue, reducedBuf, CL_TRUE, 0,
                               sizeof(float) * reducedTiles.size(),
                               reducedTiles.data(), 0, nullptr, nullptr));

  std::vector<float> groupedY =
      unpackTaskTilesFp32(reducedTiles, tasks, meta.totalGroupedTokens, cfg.n);
  std::vector<float> out(cfg.tokens * cfg.n, 0.0f);
  gatherByToken(groupedY, out, cfg.n, meta);
  auto e2eEnd = std::chrono::steady_clock::now();

  int pass = 0;
  float maxAbsErr = 0.0f;
  for (size_t i = 0; i < out.size(); ++i) {
    float absErr = std::fabs(out[i] - ref[i]);
    maxAbsErr = std::max(maxAbsErr, absErr);
    pass += absErr < 2e-1f;
  }

  std::printf("bmm_fp16: experts=%d tokens=%d k=%d n=%d splitK=%d grouped=%d "
              "tasks=%zu pass=%d/%zu maxAbsErr=%e\n",
              cfg.experts, cfg.tokens, cfg.k, cfg.n, cfg.splitKSlices,
              meta.totalGroupedTokens, tasks.size(), pass, out.size(), maxAbsErr);
  double flops = effectiveFlops(cfg);
  double e2eNs = std::chrono::duration<double, std::nano>(e2eEnd - e2eStart).count();
  std::printf("bmm_fp16 kernel: ms=%.3f gflops=%.6f\n", msFromNs(kernelNs),
              gflopsFromNs(flops, kernelNs));
  std::printf("bmm_fp16 e2e: ms=%.3f gflops=%.6f\n", msFromNs(e2eNs),
              gflopsFromNs(flops, e2eNs));

  clReleaseMemObject(xBuf);
  clReleaseMemObject(routeBuf);
  clReleaseMemObject(slotBuf);
  clReleaseMemObject(offsetsBuf);
  clReleaseMemObject(aTileBuf);
  clReleaseMemObject(groupedToTokenBuf);
  clReleaseMemObject(weightBuf);
  clReleaseMemObject(taskExpertBuf);
  clReleaseMemObject(taskGroupedBuf);
  clReleaseMemObject(taskNTileBuf);
  clReleaseMemObject(sliceBeginBuf);
  clReleaseMemObject(sliceEndBuf);
  clReleaseMemObject(partialBuf);
  clReleaseMemObject(reducedBuf);
  clReleaseKernel(scatterKernel);
  clReleaseKernel(gemmKernel);
  clReleaseKernel(reduceKernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(ctx.queue);
  clReleaseContext(ctx.context);
  return pass == int(out.size()) ? 0 : 1;
}
