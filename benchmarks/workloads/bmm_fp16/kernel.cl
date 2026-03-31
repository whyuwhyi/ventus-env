__kernel void route_scatter_top1(__global const uint *x_packed,
                                 __global const int *route,
                                 __global const int *slot,
                                 __global const int *offsets,
                                 int tokens,
                                 int xWords,
                                 int kTiles,
                                 __global uint *a_tiles,
                                 __global int *grouped_to_token) {
  int gid = get_global_id(0);
  int total = tokens * xWords;
  if (gid >= total)
    return;

  int token = gid / xWords;
  int word = gid % xWords;
  int expert = route[token];
  int groupedToken = offsets[expert] + slot[token];
  int groupedTile = groupedToken >> 4;
  int rowInTile = groupedToken & 15;
  int kTile = word >> 3;
  int wordInTile = word & 7;
  int linearWord = rowInTile * 8 + wordInTile;
  int reg = linearWord >> 5;
  int lane = linearWord & 31;
  int base = ((groupedTile * kTiles + kTile) * 32 + lane) * 4;
  a_tiles[base + reg] = x_packed[token * xWords + word];
  if (word == 0)
    grouped_to_token[groupedToken] = token;
}

__kernel void grouped_gemm_splitk_fp16(__global const uint *a_tiles,
                                       __global const uint *b_tiles,
                                       __global const int *task_expert,
                                       __global const int *task_grouped_base,
                                       __global const int *task_n_tile,
                                       __global const int *slice_k_begin,
                                       __global const int *slice_k_end,
                                       int taskOffset,
                                       int launchTaskCount,
                                       int splitKSlices,
                                       int kTiles,
                                       int nTiles,
                                       __global float *partial_tiles) {
  int task = get_group_id(0);
  int tid = get_local_id(0);
  int totalTasks = launchTaskCount * splitKSlices;
  if (task >= totalTasks)
    return;

  int baseTask = taskOffset + task / splitKSlices;
  int slice = task % splitKSlices;
  int globalTask = taskOffset * splitKSlices + task;
  int expert = task_expert[baseTask];
  int groupedTile = task_grouped_base[baseTask] >> 4;
  int nTile = task_n_tile[baseTask];
  int kTileBegin = slice_k_begin[slice];
  int kTileEnd = slice_k_end[slice];
  float8 partial = (float8)(0.0f);

  for (int kTile = kTileBegin; kTile < kTileEnd; ++kTile) {
    int aBase = ((groupedTile * kTiles + kTile) * 32 + tid) * 4;
    int bBase = ((((expert * nTiles) + nTile) * kTiles + kTile) * 32 + tid) * 4;
    uint4 a = (uint4)(a_tiles[aBase + 0], a_tiles[aBase + 1], a_tiles[aBase + 2],
                      a_tiles[aBase + 3]);
    uint4 b = (uint4)(b_tiles[bBase + 0], b_tiles[bBase + 1], b_tiles[bBase + 2],
                      b_tiles[bBase + 3]);
    partial =
        __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32(a, b, partial);
  }

  ((__global float8 *)partial_tiles)[globalTask * 32 + tid] = partial;
}

__kernel void reduce_splitk_fp32(__global const float *partial_tiles,
                                 int taskOffset,
                                 int launchTaskCount,
                                 int splitKSlices,
                                 __global float *reduced_tiles) {
  int localTask = get_group_id(0);
  int tid = get_local_id(0);
  if (localTask >= launchTaskCount)
    return;

  int baseTask = taskOffset + localTask;
  float8 acc = (float8)(0.0f);
  for (int slice = 0; slice < splitKSlices; ++slice)
    acc += ((__global const float8 *)partial_tiles)[(baseTask * splitKSlices + slice) * 32 + tid];
  ((__global float8 *)reduced_tiles)[baseTask * 32 + tid] = acc;
}
