__kernel void gelu_fp32(__global const float *input, __global float *output) {
  int tid = get_global_id(0);
  output[tid] = __builtin_riscv_ventus_vgelu_approx_f32(input[tid]);
}
