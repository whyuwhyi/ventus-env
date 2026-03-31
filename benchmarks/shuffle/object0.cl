__kernel void shuffle_down_i32_masked(__global const uint *input, __global uint *output) {
  int tid = get_global_id(0);
  if (tid < 16)
    output[tid] = __builtin_riscv_ventus_shuffle_down_i32(input[tid], 1);
  else
    output[tid] = input[tid];
}
