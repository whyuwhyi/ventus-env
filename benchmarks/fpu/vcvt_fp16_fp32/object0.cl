__kernel void vcvt_fp16_fp32(__global const float *input, __global uint *output) {
  int tid = get_global_id(0);
  output[tid] = __builtin_riscv_ventus_vcvt_fp16_fp32(input[tid]);
}
