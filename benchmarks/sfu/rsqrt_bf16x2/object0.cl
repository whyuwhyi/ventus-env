__kernel void rsqrt_bf16x2(__global const uint *input, __global uint *output) {
  int tid = get_global_id(0);
  uint value = input[tid];
  output[tid] = __builtin_riscv_ventus_vrsqrt_approx_bf16x2(value);
}
