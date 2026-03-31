__kernel void rcp_fp16x2(__global const uint *input, __global uint *output) {
  int tid = get_global_id(0);
  uint value = input[tid];
  output[tid] = __builtin_riscv_ventus_vrcp_approx_f16x2(value);
}
