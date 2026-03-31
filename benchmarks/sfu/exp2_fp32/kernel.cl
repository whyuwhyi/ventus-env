__kernel void exp2_fp32(__global const float *input, __global float *output) {
  int tid = get_global_id(0);
  float value = input[tid];
  output[tid] = __builtin_riscv_ventus_vex2_approx_f32(value);
}
