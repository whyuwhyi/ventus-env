__kernel void vcvt_fp32_fp16(__global const uint *input, __global float *output) {
  int tid = get_global_id(0);
  output[tid] = __builtin_riscv_ventus_vcvt_fp32_fp16(input[tid]);
}
