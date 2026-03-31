__kernel void shuffle_up_i32(__global const uint *input, __global uint *output) {
  int tid = get_global_id(0);
  output[tid] = __builtin_riscv_ventus_shuffle_up_i32(input[tid], 1);
}
