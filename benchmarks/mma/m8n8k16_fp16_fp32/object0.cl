__kernel void mma_m8n8k16_fp16_fp32(
    __global const uint *a_window,
    __global const uint *b_window,
    __global const float *c_window,
    __global float *d_window) {
  int tid = get_global_id(0);
  int a_base = tid * 2;
  int b_base = tid * 2;
  int c_base = tid * 2;

  uint a0 = a_window[a_base + 0];
  uint a1 = a_window[a_base + 1];
  uint b0 = b_window[b_base + 0];
  uint b1 = b_window[b_base + 1];
  float c0 = c_window[c_base + 0];
  float c1 = c_window[c_base + 1];

  uint2 a = (uint2)(a0, a1);
  uint2 b = (uint2)(b0, b1);
  float2 c = (float2)(c0, c1);

  float2 d = __builtin_riscv_ventus_mma_m8n8k16_row_col_f32_f16_f16_f32(a, b, c);
  ((__global float2 *)d_window)[tid] = d;
}
